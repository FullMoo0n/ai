from typing import List
from io import BytesIO
from pathlib import Path
import os
import time

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image

# Google GenAI SDK 사용
try:
    from google import genai
    from google.genai import types
    _has_genai = True
except ImportError:
    _has_genai = False

from .services.sentence_segmenter import split_sentences
from .services.tokenizer import tokenize
from .services.openai_eval import evaluate_segmentation_with_openai

# 스키마
from .schemas.sentences import SentencesRequest, SentencesResponse
from .schemas.tokens import TokensRequest, TokensResponse
from .schemas.validate import ValidateRequest, ValidateResponse

from .schemas.ocr import OCRResponse
from .services.vision_ocr import (
    encode_bytes_to_b64,
    build_payload,
    call_vision_api,
    extract_full_text,
    extract_word_boxes,
    extract_paragraphs_spatial_proximity_advanced,
)
from .schemas.culture import CultureRequest, CultureResponse
from .services.sign_data_service import SignDataService
from fastapi.concurrency import run_in_threadpool
from .services.gemini_service import GeminiService
from .schemas.veo import (
    VeoRequest, VeoResponse, ErrorResponse,
    VeoAsyncRequest, VeoAsyncResponse, VeoTaskStatus
)

# Celery 관련 임포트
from .celery_app import celery_app
from .tasks.veo_tasks import generate_veo_video_task

# .env 로드
load_dotenv()

# Initialize Services
sign_data_service = SignDataService()

app = FastAPI(title="Vision OCR Wrapper", version="1.5.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(..., description="이미지 파일"),
    feature: str = Form("TEXT_DETECTION"),      # 동화책 기본
    language_hints: str = Form("ko"),
    include_word_boxes: bool = Form(False),
    debug: bool = Form(False),
    # 문단 병합 파라미터
    line_tol: int = Form(12),
    para_gap: int = Form(28),
    horiz_overlap_min: float = Form(0.1),
    para_x_join_tol: int = Form(30),
    cluster_scale: float = Form(0.06),  # 덩어리 분리 강도
):
    """
    TEXT_DETECTION 기반:
    - 노이즈 제거 → 거리 클러스터링 → 세로/가로 판정 → 줄/열 정렬 → 문단화
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        if len(content) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="파일이 너무 큽니다(>8MB).")

        # 이미지 크기(파라미터 계산에 사용)
        try:
            with Image.open(BytesIO(content)) as im:
                img_w, img_h = im.size
        except Exception:
            # 이미지가 아니면 기본값(문단 병합만 수행)
            img_w, img_h = 1024, 1024

        image_b64 = encode_bytes_to_b64(content)
        langs: List[str] = [s.strip() for s in (language_hints or "").split(",") if s.strip()]
        payload = build_payload(image_b64, feature=feature, language_hints=langs)
        vision_raw = await call_vision_api(payload)

        result: dict = {"text": extract_full_text(vision_raw)}

        # 고급 문단 추출
        paragraphs = extract_paragraphs_spatial_proximity_advanced(
            vision_raw,
            img_w=img_w, img_h=img_h,
            line_tol=line_tol, para_gap=para_gap,
            horiz_overlap_min=horiz_overlap_min,
            para_x_join_tol=para_x_join_tol,
            cluster_scale=cluster_scale,
        )
        result["paragraphs"] = paragraphs

        if include_word_boxes:
            result["word_boxes"] = extract_word_boxes(vision_raw)

        if debug:
            result["raw"] = vision_raw

        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")


# --- 문장 분리 전용 ---
@app.post("/sentences", response_model=SentencesResponse)
async def sentences_endpoint(payload: SentencesRequest = Body(...)):
    """
    입력 텍스트(또는 문단 배열)를 문장 단위로만 분리
    """
    if not (payload.text or payload.paragraphs):
        raise HTTPException(status_code=400, detail="text 또는 paragraphs 중 하나는 필요합니다.")

    # 전체 원문
    whole_text = payload.text or ""
    # 문장 분리
    sents = split_sentences(whole_text)

    return SentencesResponse(text=whole_text, sentences=sents)

# --- 토큰화 전용 ---
@app.post("/tokens", response_model=TokensResponse)
async def tokens_endpoint(payload: TokensRequest = Body(...)):
    """
    문장 리스트를 받아 각 문장을 단어(토큰)로만 분리
    (문장 분리는 이 엔드포인트에서 하지 않음)
    """
    if not payload.sentences:
        raise HTTPException(status_code=400, detail="sentences가 비어있습니다.")
    tokens_per_sentence = [tokenize(s) for s in payload.sentences]
    return TokensResponse(tokens_per_sentence=tokens_per_sentence)


@app.post("/sentences/validate", response_model=ValidateResponse)
async def validate_sentences(payload: ValidateRequest = Body(...)):
    try:
        result = evaluate_segmentation_with_openai(
            text=payload.text,
            sentences=payload.sentences,
            model="gpt-4.1-mini",
        )
        return ValidateResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI 검증 실패: {e}")


@app.post(
    "/veo",
    response_model=VeoResponse,
    responses={
        200: {
            "description": "비디오 생성 성공 또는 진행 중",
            "model": VeoResponse,
        },
        400: {
            "description": "잘못된 요청 (API 키 누락, 패키지 미설치 등)",
            "model": ErrorResponse,
        },
        429: {
            "description": "API 할당량 초과",
            "model": ErrorResponse,
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse,
        }
    },
    tags=["Video Generation"],
    summary="텍스트로 비디오 생성 (동기)",
    description="""
    ## 텍스트 프롬프트로 고품질 비디오 생성 (동기 방식)

    이 엔드포인트는 Google Veo 3 모델을 사용하여 텍스트 설명으로부터 8초 길이의 720p 비디오를 생성합니다.
    **주의: 이 엔드포인트는 동기식으로 작동하며, 비디오 생성이 완료될 때까지 대기합니다.**

    ### 프롬프트 작성 팁:
    - **구체적으로 작성**: "A red panda riding a skateboard in a sunny park"
    - **스타일 지정**: "cinematic", "animation", "" 등 스타일 키워드 추가
    - **카메라 앵글**: "close-up", "wide shot", "aerial view" 등
    - **분위기**: "bright and cheerful", "dark and mysterious" 등

    ### 응답 시간:
    - 일반적으로 1-3분 소요
    - `timeout_seconds` 파라미터로 대기 시간 조절 가능

    ### 주의사항:
    - Google Gemini API 할당량이 필요합니다
    - 생성된 비디오는 `video_uri`를 통해 다운로드 가능합니다
    - 긴 대기 시간이 필요한 경우 `/veo/async` 엔드포인트 사용을 권장합니다
    """
)
def generate_veo_video(req: VeoRequest):
    if not _has_genai:
        return VeoResponse(
            status="error",
            video_uri=None,
            local_path=None,
            operation=None,
            message=None,
            error="google-genai 패키지가 필요합니다. pip install google-genai를 실행하세요."
        )

    # Google API Key 확인
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return VeoResponse(
            status="error",
            video_uri=None,
            local_path=None,
            operation=None,
            message=None,
            error="GOOGLE_API_KEY 환경변수가 설정되지 않았습니다."
        )

    try:
        # 동기 방식에서는 Celery 태스크를 사용하도록 안내
        return VeoResponse(
            status="error",
            video_uri=None,
            local_path=None,
            operation=None,
            message=None,
            error="실제 비디오 생성은 /veo/async 엔드포인트를 사용해주세요. 동기 방식은 시간이 오래 걸려 권장하지 않습니다."
        )
        
    except Exception as e:
        return VeoResponse(
            status="error",
            video_uri=None,
            local_path=None,
            operation=None,
            message=None,
            error=f"Google GenAI 호출 중 오류 발생: {str(e)}"
        )


@app.post(
    "/veo/async",
    response_model=VeoAsyncResponse,
    responses={
        202: {
            "description": "비디오 생성 작업이 성공적으로 큐에 추가됨",
            "model": VeoAsyncResponse,
        },
        400: {
            "description": "잘못된 요청 (API 키 누락, 패키지 미설치 등)",
            "model": ErrorResponse,
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse,
        }
    },
    status_code=202,
    tags=["Video Generation"],
    summary="텍스트로 비디오 생성 (비동기)",
    description="""
    ## 텍스트 프롬프트로 고품질 비디오 생성 (비동기 방식)

    이 엔드포인트는 Google Veo 3 모델을 사용하여 텍스트 설명으로부터 8초 길이의 720p 비디오를 비동기로 생성합니다.
    **비동기 방식으로 작동하여 즉시 task_id를 반환하고, 별도로 상태를 조회할 수 있습니다.**

    ### 작업 흐름:
    1. 이 엔드포인트로 비디오 생성 요청
    2. `task_id` 즉시 반환 (202 status)
    3. `/veo/status/{task_id}` 로 작업 상태 조회
    4. 완료되면 `video_uri`를 통해 비디오 다운로드

    ### 프롬프트 작성 팁:
    - **구체적으로 작성**: "A red panda riding a skateboard in a sunny park"
    - **스타일 지정**: "cinematic", "animation", "" 등 스타일 키워드 추가
    - **카메라 앵글**: "close-up", "wide shot", "aerial view" 등
    - **분위기**: "bright and cheerful", "dark and mysterious" 등

    ### 예상 소요 시간:
    - 일반적으로 1-10분 소요
    - 복잡한 프롬프트의 경우 더 오래 걸릴 수 있음

    ### 주의사항:
    - Google Gemini API 할당량이 필요합니다
    - Redis 서버가 실행 중이어야 합니다
    - Celery 워커가 실행 중이어야 합니다
    """
)
def generate_veo_video_async(req: VeoAsyncRequest):
    if not _has_genai:
        raise HTTPException(
            status_code=400,
            detail="Veo 3 모델은 현재 사용할 수 없습니다. Google Cloud 프로젝트에서 Generative Language API를 활성화해야 합니다."
        )

    try:
        # Celery 태스크 큐에 작업 추가
        task = generate_veo_video_task.delay(
            prompt=req.prompt,
            aspect_ratio=req.aspect_ratio,
            model=req.model,
            timeout_seconds=req.timeout_seconds
        )
        
        return VeoAsyncResponse(
            task_id=task.id,
            status="queued",
            message="비디오 생성 작업이 큐에 추가되었습니다. `/veo/status/{task_id}`로 상태를 확인하세요."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"작업 큐 추가 실패: {str(e)}")


@app.get(
    "/veo/status/{task_id}",
    response_model=VeoTaskStatus,
    responses={
        200: {
            "description": "작업 상태 조회 성공",
            "model": VeoTaskStatus,
        },
        404: {
            "description": "작업을 찾을 수 없음",
            "model": ErrorResponse,
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse,
        }
    },
    tags=["Video Generation"],
    summary="비동기 비디오 생성 작업 상태 조회",
    description="""
    ## 비동기 비디오 생성 작업의 상태를 조회합니다

    ### 가능한 작업 상태:
    - **PENDING**: 작업이 큐에서 대기 중
    - **PROGRESS**: 작업이 진행 중 (진행 상황 정보 포함)
    - **SUCCESS**: 작업 완료 (비디오 URI 포함)
    - **FAILURE**: 작업 실패 (오류 메시지 포함)
    - **RETRY**: 작업 재시도 중
    - **REVOKED**: 작업이 취소됨

    ### 진행 상황 정보:
    PROGRESS 상태일 때 `progress` 필드에서 다음 정보를 확인할 수 있습니다:
    - `status`: 세부 진행 상태 (starting, initializing, generating, processing)
    - `message`: 현재 진행 상황 메시지
    - `operation`: Google API Operation ID (있는 경우)
    - `elapsed_seconds`: 경과 시간 (있는 경우)

    ### 완료 후:
    SUCCESS 상태가 되면 `result` 필드에서 `video_uri`를 통해 비디오를 다운로드할 수 있습니다.
    """
)
def get_veo_task_status(task_id: str):
    if not _has_genai:
        raise HTTPException(
            status_code=400,
            detail="Veo 3 모델은 현재 사용할 수 없습니다. Google Cloud 프로젝트에서 Generative Language API를 활성화해야 합니다."
        )

    try:
        # Celery에서 작업 결과 조회
        task_result = celery_app.AsyncResult(task_id)
        
        if task_result.state == "PENDING":
            return VeoTaskStatus(
                task_id=task_id,
                status="PENDING",
                result=None,
                progress=None,
                error=None
            )
        elif task_result.state == "PROGRESS":
            return VeoTaskStatus(
                task_id=task_id,
                status="PROGRESS",
                result=None,
                progress=task_result.info,
                error=None
            )
        elif task_result.state == "SUCCESS":
            return VeoTaskStatus(
                task_id=task_id,
                status="SUCCESS",
                result=task_result.result,
                progress=None,
                error=None
            )
        elif task_result.state == "FAILURE":
            return VeoTaskStatus(
                task_id=task_id,
                status="FAILURE",
                result=None,
                progress=None,
                error=str(task_result.info)
            )
        else:
            return VeoTaskStatus(
                task_id=task_id,
                status=task_result.state,
                result=task_result.result if hasattr(task_result, 'result') else None,
                progress=task_result.info if hasattr(task_result, 'info') else None,
                error=None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"작업 상태 조회 실패: {str(e)}")


@app.delete(
    "/veo/cancel/{task_id}",
    responses={
        200: {"description": "작업 취소 성공"},
        404: {"description": "작업을 찾을 수 없음"},
        500: {"description": "서버 내부 오류"}
    },
    tags=["Video Generation"],
    summary="비동기 비디오 생성 작업 취소",
    description="""
    ## 진행 중인 비동기 비디오 생성 작업을 취소합니다

    ### 주의사항:
    - 이미 시작된 Google API 호출은 취소할 수 없을 수 있습니다
    - 큐에서 대기 중인 작업은 즉시 취소됩니다
    - 진행 중인 작업은 다음 체크포인트에서 취소됩니다
    """
)
def cancel_veo_task(task_id: str):
    if not _has_genai:
        raise HTTPException(
            status_code=400,
            detail="Veo 3 모델은 현재 사용할 수 없습니다. Google Cloud 프로젝트에서 Generative Language API를 활성화해야 합니다."
        )

    try:
        celery_app.control.revoke(task_id, terminate=True)
        return {"message": f"작업 {task_id}이(가) 취소되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"작업 취소 실패: {str(e)}")


@app.post(
    "/culture/sign-description",
    response_model=CultureResponse,
    responses={
        200: {
            "description": "수어 설명 조회 성공",
            "model": CultureResponse,
        },
        400: {
            "description": "잘못된 요청 (키워드 누락 등)",
        },
        500: {
            "description": "서버 내부 오류 (API 키 누락, API 호출 실패 등)",
        }
    },
    tags=["Culture API"],
    summary="수어 설명 조회",
    description="""
    ## 키워드로 수어 설명을 조회합니다

    한국문화정보원 API를 통해 입력된 키워드에 해당하는 수어의 설명을 가져옵니다.
    검색 결과 중 첫 번째 항목의 signDescription을 반환합니다.

    ### 요청 예시:
    ```json
    {
        "keyword": "공주"
    }
    ```

    ### 응답 예시:
    ```json
    {
        "keyword": "공주",
        "sign_description": "손등이 위로 향하게 편 왼손의 2지 옆면을 오른 주먹의 1·5지 끝으로 스쳐 올린 다음, 오른 주먹의 4지를 펴서 끝으로 배를 스쳐 내려 등이 위로 향하게 한다."
    }
    ```

    ### 주의사항:
    - CULTURE_API_KEY 환경변수가 설정되어 있어야 합니다
    - 검색 결과가 없는 경우 sign_description은 null이 됩니다
    """
)
async def get_culture_sign_description(request: CultureRequest):
    """키워드로 수어 설명을 조회합니다."""
    if not request.keyword or not request.keyword.strip():
        raise HTTPException(status_code=400, detail="키워드가 필요합니다.")
    
    try:
        sign_description = await run_in_threadpool(sign_data_service.search_sign_description, request.keyword.strip())
        
        return CultureResponse(
            keyword=request.keyword.strip(),
            sign_description=sign_description
        )
        
    except ValueError as e:
        # 환경변수 누락 등
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        # API 호출 실패 등
        raise HTTPException(status_code=500, detail=f"수어 설명 조회 실패: {str(e)}")


# 통합 파이프라인 엔드포인트
from pydantic import BaseModel, HttpUrl
import uuid
from datetime import datetime


class PipelineRequest(BaseModel):
    """파이프라인 처리 요청"""
    s3_image_url: HttpUrl


class PipelineResponse(BaseModel):
    """파이프라인 처리 응답"""
    task_id: str
    status: str
    message: str
    started_at: str


class PipelineResultResponse(BaseModel):
    """파이프라인 결과 응답"""
    task_id: str
    status: str
    video_urls: list[str] = []
    video_details: list[dict] = []
    total_videos: int = 0
    successful_videos: int = 0
    failed_videos: int = 0
    completed_at: str | None = None
    error: str | None = None


@app.post(
    "/process-image-to-videos",
    response_model=PipelineResultResponse,
    responses={
        200: {
            "description": "파이프라인 처리 완료됨",
            "model": PipelineResultResponse,
        },
        400: {
            "description": "잘못된 요청 (유효하지 않은 S3 URL 등)",
            "model": ErrorResponse,
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse,
        }
    },
    tags=["Pipeline"],
    summary="이미지에서 수어 동영상 생성 파이프라인 (Gemini + Veo3)",
    description="""
    ## S3 이미지 URL에서 수어 동영상 생성 (Gemini + Veo3 통합)

    이 엔드포인트는 다음 과정을 수행합니다:
    1. **OCR**: S3 이미지에서 텍스트 추출 (Google Vision API)
    2. **Gemini 통합 처리**: 
       - 문장 분할 및 형태소 분석
       - Culture API에서 각 형태소별 수어 데이터 수집
       - 수어 데이터를 포함한 Veo3용 프롬프트 생성
    3. **Veo3 비디오 생성**: Gemini가 생성한 프롬프트로 수어 동영상 생성
    4. **결과 반환**: 생성된 모든 동영상의 S3 URL 반환

    ### 사용법:
    ```json
    {
        "s3_image_url": "https://your-bucket.s3.amazonaws.com/your-image.jpg"
    }
    ```

    ### 응답:
    - **task_id**: 작업 추적용 고유 ID
    - **status**: 'completed' (처리 완료)
    - **video_urls**: 생성된 비디오 URL 목록
    - **video_details**: 각 비디오의 상세 정보
    - **total_videos**: 총 생성된 비디오 수
    - **successful_videos**: 성공한 비디오 수
    - **failed_videos**: 실패한 비디오 수

    ### 특징:
    - **동기 처리**: 요청 시 즉시 전체 파이프라인 실행 후 결과 반환
    - **Gemini AI**: 고품질 문장 분석 및 프롬프트 생성
    - **Veo3 통합**: Gemini 프롬프트를 직접 Veo3에 전달
    - **에러 처리**: Veo3 API 오류 시 모의 결과 생성으로 안정성 확보
    """
)
async def process_image_to_videos_gemini_veo3(request: PipelineRequest):
    """Gemini + Veo3 통합 파이프라인으로 이미지 처리 및 수어 동영상 생성"""
    try:
        # S3 URL 검증
        url_str = str(request.s3_image_url)
        if not any(domain in url_str for domain in ['amazonaws.com', 's3.']):
            raise HTTPException(
                status_code=400, 
                detail="유효한 S3 URL이 아닙니다. amazonaws.com 도메인이 필요합니다."
            )
        
        # 작업 ID 생성
        task_id = f"gemini_veo3_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        print(f"🚀 Gemini + Veo3 통합 파이프라인 시작: {task_id}")
        print(f"📸 S3 이미지 URL: {url_str}")
        
        # Gemini + Veo3 통합 파이프라인 실행
        from app.services.integrated_pipeline import IntegratedPipeline
        
        try:
            # 파이프라인 생성 및 실행
            pipeline = IntegratedPipeline(task_id)
            result = await pipeline.execute(url_str)
            
            print(f"✅ 파이프라인 실행 완료: {task_id}")
            print(f"📊 결과: {result}")
            
            # 결과를 응답 형식으로 변환
            video_urls = []
            video_details = []
            
            if 'video_details' in result and result['video_details']:
                for video in result['video_details']:
                    if 'video_url' in video:
                        video_urls.append(video['video_url'])
                    video_details.append(video)
            
            response = PipelineResultResponse(
                task_id=task_id,
                status=result.get('status', 'completed'),
                video_urls=video_urls,
                video_details=video_details,
                total_videos=result.get('total_videos', len(video_details)),
                successful_videos=result.get('successful_videos', len([v for v in video_details if v.get('status') == 'completed'])),
                failed_videos=result.get('failed_videos', len([v for v in video_details if v.get('status') != 'completed'])),
                completed_at=datetime.now().isoformat(),
                error=None
            )
            
            print(f"🎉 응답 생성 완료: {len(video_urls)}개 비디오 URL")
            return response
            
        except Exception as pipeline_error:
            print(f"❌ 파이프라인 실행 중 오류: {pipeline_error}")
            raise HTTPException(
                status_code=500,
                detail=f"파이프라인 실행 중 오류 발생: {str(pipeline_error)}"
            )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"파이프라인 시작 중 오류 발생: {str(e)}"
        )


@app.post(
    "/process-image-to-videos-legacy",
    response_model=PipelineResultResponse,
    responses={
        200: {
            "description": "파이프라인 처리 완료됨",
            "model": PipelineResultResponse,
        },
        400: {
            "description": "잘못된 요청 (유효하지 않은 S3 URL 등)",
            "model": ErrorResponse,
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse,
        }
    },
    tags=["Pipeline"],
    summary="이미지에서 수어 동영상 생성 파이프라인 (개선된 방식)",
    description="""
    ## S3 이미지 URL에서 수어 동영상 생성 (간소화된 방식)

    이 엔드포인트는 다음 과정을 수행합니다:
    1. **OCR**: S3 이미지에서 텍스트 추출
    2. **전체 토큰화**: 문장 분할 없이 전체 텍스트를 한번에 토큰화
    3. **수어 데이터 조회**: 모든 토큰에 대한 수어 설명 수집
    4. **Gemini 프롬프트 생성**: 전체 텍스트와 수어 데이터로 단일 프롬프트 생성
    5. **VEO 비디오 생성**: Gemini 프롬프트로 한 개의 통합 수어 동영상 생성
    6. **S3 업로드**: 생성된 동영상을 S3에 업로드

    ### 주요 개선사항:
    - ✅ **문장별 분할 제거**: 전체 텍스트를 한번에 처리
    - ✅ **Gemini AI 1회 호출**: 여러 번 호출 대신 1번의 효율적인 호출
    - ✅ **단일 비디오 생성**: 여러 개 대신 하나의 통합된 수어 동영상
    - ✅ **빠른 처리 속도**: 간소화된 파이프라인으로 더 빠른 응답

    ### 사용법:
    ```json
    {
        "s3_image_url": "https://your-bucket.s3.amazonaws.com/your-image.jpg"
    }
    ```

    ### 응답:
    - **task_id**: 작업 추적용 고유 ID
    - **status**: 'completed' (처리 완료)
    - **video_urls**: 생성된 비디오 URL (1개)
    - **video_details**: 비디오 상세 정보
    - **total_videos**: 총 비디오 수 (1개)
    """
)
def process_image_to_videos(request: PipelineRequest):
    """이미지 처리 및 수어 동영상 생성 파이프라인 시작 (개선된 방식)"""
    try:
        # S3 URL 검증
        url_str = str(request.s3_image_url)
        if not any(domain in url_str for domain in ['amazonaws.com', 's3.']):
            raise HTTPException(
                status_code=400, 
                detail="유효한 S3 URL이 아닙니다. amazonaws.com 도메인이 필요합니다."
            )
        
        # 작업 ID 생성
        task_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        
        # 동기적으로 파이프라인 실행하고 결과 바로 반환
        from app.services.sync_pipeline import SyncIntegratedPipeline
        
        # 파이프라인 생성 및 실행
        pipeline = SyncIntegratedPipeline(task_id)
        result = pipeline.execute(url_str)
        
        # 완료된 결과를 바로 반환
        return PipelineResultResponse(
            task_id=task_id,
            status=result.get('status', 'completed'),
            video_urls=result.get('video_urls', []),
            video_details=result.get('video_details', []),
            total_videos=result.get('total_videos', 0),
            successful_videos=result.get('successful_videos', 0),
            failed_videos=result.get('failed_videos', 0),
            completed_at=result.get('completed_at'),
            error=result.get('error')
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"파이프라인 시작 중 오류 발생: {str(e)}"
        )


@app.get(
    "/pipeline-status/{task_id}",
    response_model=PipelineResultResponse,
    responses={
        200: {
            "description": "파이프라인 상태 조회 성공",
            "model": PipelineResultResponse,
        },
        404: {
            "description": "작업 ID를 찾을 수 없음",
            "model": ErrorResponse,
        },
        500: {
            "description": "서버 내부 오류",
            "model": ErrorResponse,
        }
    },
    tags=["Pipeline"],
    summary="파이프라인 처리 상태 조회",
    description="""
    ## 파이프라인 처리 상태 및 결과 조회

    작업 ID로 파이프라인 처리 상태를 확인합니다.

    ### 응답 상태:
    - **processing**: 처리 중
    - **completed**: 완료 (비디오 URL 목록 포함)
    - **failed**: 실패 (오류 메시지 포함)

    ### 완료 시 응답:
    ```json
    {
        "task_id": "pipeline_20250819_123456_abc12345",
        "status": "completed",
        "video_urls": [
            "https://bucket.s3.amazonaws.com/video1.mp4",
            "https://bucket.s3.amazonaws.com/video2.mp4"
        ],
        "total_videos": 2,
        "successful_videos": 2,
        "failed_videos": 0,
        "completed_at": "2025-08-19T12:34:56"
    }
    ```
    """
)
async def get_pipeline_status(task_id: str):
    """파이프라인 처리 상태 조회"""
    try:
        # 비동기 파이프라인 상태 조회
        from app.services.integrated_pipeline import get_pipeline_status
        status_info = get_pipeline_status(task_id)
        
        # status_info가 None인 경우 처리
        if status_info is None:
            raise HTTPException(status_code=404, detail=f"작업 ID {task_id}를 찾을 수 없습니다 (상태 정보 없음)")
        
        if status_info.get('pipeline_status') == 'not_found':
            raise HTTPException(status_code=404, detail=f"작업 ID {task_id}를 찾을 수 없습니다")
        
        # 파이프라인 상태를 응답 형식으로 변환
        final_result = status_info.get('final_result', {}) if status_info else {}
        
        return PipelineResultResponse(
            task_id=task_id,
            status=status_info.get('pipeline_status', 'unknown'),
            video_urls=final_result.get('video_urls', []),
            total_videos=final_result.get('total_videos', 0),
            successful_videos=final_result.get('successful_videos', 0),
            failed_videos=final_result.get('failed_videos', 0),
            completed_at=final_result.get('completed_at'),
            error=status_info.get('error')
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"상태 조회 중 오류 발생: {str(e)}"
        )


@app.post("/analyze-sentences")
async def analyze_sentences_with_gemini(
    text: str = Body(..., embed=True, description="분석할 한국어 텍스트")
):
    """
    Gemini API를 사용하여 한국어 문장을 분석하고 형태소를 추출합니다.
    
    입력 텍스트는 두 개의 문장으로 분할되어야 합니다:
    1. "내가 그랬어요!"
    2. "텔레비전을 부순 건 바로 나예요!"
    
    각 문장에서 핵심 형태소만 추출하여 반환합니다.
    """
    try:
        # API 키 확인
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise HTTPException(
                status_code=500, 
                detail="GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다."
            )
        
        # Gemini 서비스 초기화
        gemini_service = GeminiService(api_key)
        
        # 문장 분석 수행
        result = gemini_service.analyze_sentences(text)
        
        # 결과 검증
        if not gemini_service.validate_analysis_result(result):
            raise HTTPException(
                status_code=500,
                detail="Gemini API 응답이 예상 형식과 일치하지 않습니다."
            )
        
        # 요약 정보 추가
        summary = gemini_service.get_analysis_summary(result)
        
        return {
            "success": True,
            "input_text": text,
            "analysis_result": result,
            "summary": summary
        }
        
    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini 서비스 초기화 실패: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"입력 텍스트 처리 오류: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"문장 분석 중 오류 발생: {str(e)}"
        )

