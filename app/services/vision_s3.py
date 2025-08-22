import logging
from typing import Dict, Any, Optional, List
from .s3_utils import (
    validate_s3_url_format, 
    check_s3_object_exists, 
    get_s3_public_url,
    S3InvalidURLError, 
    S3AccessError
)
from .vision_ocr import (
    build_payload,
    call_vision_api,
    extract_full_text,
    extract_word_boxes,
    extract_paragraphs_spatial_proximity_advanced
)

logger = logging.getLogger(__name__)


class VisionS3Error(Exception):
    """Vision S3 처리 관련 에러 기본 클래스"""
    pass


class VisionS3URLError(VisionS3Error):
    """S3 URL 관련 에러"""
    pass


class VisionS3APIError(VisionS3Error):
    """Google Vision API 호출 에러"""
    pass


async def process_s3_image_with_vision(
    s3_url: str,
    feature: str = "TEXT_DETECTION",
    language_hints: Optional[List[str]] = None,
    include_word_boxes: bool = False,
    # 문단 병합 파라미터
    line_tol: int = 12,
    para_gap: int = 28,
    horiz_overlap_min: float = 0.1,
    para_x_join_tol: int = 30,
    cluster_scale: float = 0.06,
    debug: bool = False
) -> Dict[str, Any]:
    """S3 이미지 URL을 Google Vision API에 직접 전달하여 OCR 처리
    
    Args:
        s3_url: 처리할 S3 이미지 URL
        feature: Vision API 기능 (기본값: TEXT_DETECTION)
        language_hints: 언어 힌트 목록 (기본값: ["ko"])
        include_word_boxes: 단어 박스 정보 포함 여부
        line_tol: 줄 간격 허용 오차
        para_gap: 문단 간격
        horiz_overlap_min: 수평 겹침 최소값
        para_x_join_tol: 문단 X축 결합 허용 오차
        cluster_scale: 클러스터 스케일
        debug: 디버그 정보 포함 여부
        
    Returns:
        Dict[str, Any]: OCR 처리 결과
        {
            "success": bool,
            "text": str,
            "paragraphs": List[str],
            "word_boxes": List[Dict] (선택사항),
            "raw": Dict (디버그 시),
            "s3_url": str,
            "public_url": str (생성된 경우)
        }
        
    Raises:
        VisionS3URLError: S3 URL 관련 오류
        VisionS3APIError: Vision API 호출 오류
    """
    try:
        # 1. S3 URL 형식 검증
        if not validate_s3_url_format(s3_url):
            raise VisionS3URLError(f"유효하지 않은 S3 URL 형식입니다: {s3_url}")
        
        logger.info(f"S3 이미지 OCR 처리 시작: {s3_url}")
        
        # 2. S3 객체 존재 여부 확인
        if not check_s3_object_exists(s3_url):
            raise VisionS3URLError(f"S3 객체를 찾을 수 없거나 접근할 수 없습니다: {s3_url}")
        
        # 3. 공개 URL 생성 (Google Vision API에서 접근 가능하도록)
        try:
            public_url = get_s3_public_url(s3_url, expires_in=3600)  # 1시간 유효
            logger.info(f"S3 presigned URL 생성 완료")
        except Exception as e:
            raise VisionS3URLError(f"S3 공개 URL 생성 실패: {str(e)}")
        
        # 4. Vision API 페이로드 구성 (URL 방식)
        if language_hints is None:
            language_hints = ["ko"]
            
        # URL 기반 페이로드 생성 (기존 build_payload 함수 수정 필요)
        payload = build_payload_for_url(public_url, feature=feature, language_hints=language_hints)
        
        # 5. Vision API 호출
        try:
            vision_raw = await call_vision_api(payload)
        except Exception as e:
            raise VisionS3APIError(f"Google Vision API 호출 실패: {str(e)}")
        
        # 6. 결과 처리
        result = {
            "success": True,
            "text": extract_full_text(vision_raw),
            "s3_url": s3_url,
            "public_url": public_url
        }
        
        # 문단 추출 (이미지 크기가 필요하므로 기본값 사용)
        paragraphs = extract_paragraphs_spatial_proximity_advanced(
            vision_raw,
            img_w=1024, img_h=1024,  # 실제 이미지 크기를 알 수 없으므로 기본값 사용
            line_tol=line_tol, 
            para_gap=para_gap,
            horiz_overlap_min=horiz_overlap_min,
            para_x_join_tol=para_x_join_tol,
            cluster_scale=cluster_scale,
        )
        result["paragraphs"] = paragraphs
        
        # 단어 박스 정보 포함 (선택사항)
        if include_word_boxes:
            result["word_boxes"] = extract_word_boxes(vision_raw)
        
        # 디버그 정보 포함 (선택사항)
        if debug:
            result["raw"] = vision_raw
        
        logger.info(f"S3 이미지 OCR 처리 완료: {len(result['text'])} 글자 추출")
        return result
        
    except (S3InvalidURLError, S3AccessError) as e:
        raise VisionS3URLError(str(e))
    except VisionS3URLError:
        raise
    except VisionS3APIError:
        raise
    except Exception as e:
        logger.error(f"S3 이미지 OCR 처리 중 예상치 못한 오류: {str(e)}")
        raise VisionS3Error(f"S3 이미지 처리 실패: {str(e)}")


def build_payload_for_url(image_url: str, feature: str = "TEXT_DETECTION", language_hints: Optional[List[str]] = None) -> Dict[str, Any]:
    """이미지 URL을 사용하여 Vision API 페이로드 생성
    
    Args:
        image_url: 이미지 URL (S3 presigned URL)
        feature: Vision API 기능
        language_hints: 언어 힌트 목록
        
    Returns:
        Dict[str, Any]: Vision API 요청 페이로드
    """
    if language_hints is None:
        language_hints = ["ko"]
    
    payload = {
        "requests": [
            {
                "image": {
                    "source": {
                        "imageUri": image_url
                    }
                },
                "features": [
                    {
                        "type": feature,
                        "maxResults": 1000  # 최대 결과 수
                    }
                ],
                "imageContext": {
                    "languageHints": language_hints
                }
            }
        ]
    }
    
    return payload


async def batch_process_s3_images_with_vision(
    s3_urls: List[str],
    feature: str = "TEXT_DETECTION",
    language_hints: Optional[List[str]] = None,
    max_concurrent: int = 5
) -> List[Dict[str, Any]]:
    """여러 S3 이미지를 병렬로 OCR 처리
    
    Args:
        s3_urls: 처리할 S3 이미지 URL 목록
        feature: Vision API 기능
        language_hints: 언어 힌트 목록
        max_concurrent: 최대 동시 처리 수
        
    Returns:
        List[Dict[str, Any]]: 각 이미지별 OCR 처리 결과 목록
    """
    import asyncio
    
    async def process_single_image(s3_url: str) -> Dict[str, Any]:
        """단일 이미지 처리 (에러 처리 포함)"""
        try:
            return await process_s3_image_with_vision(
                s3_url=s3_url,
                feature=feature,
                language_hints=language_hints
            )
        except Exception as e:
            logger.error(f"이미지 처리 실패: {s3_url} - {str(e)}")
            return {
                "success": False,
                "s3_url": s3_url,
                "error": str(e),
                "error_type": e.__class__.__name__
            }
    
    # 세마포어를 사용한 동시 처리 수 제한
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def limited_process(s3_url: str) -> Dict[str, Any]:
        async with semaphore:
            return await process_single_image(s3_url)
    
    # 모든 이미지 병렬 처리
    tasks = [limited_process(s3_url) for s3_url in s3_urls]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # 예외 처리된 결과들 정리
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "success": False,
                "s3_url": s3_urls[i],
                "error": str(result),
                "error_type": result.__class__.__name__
            })
        else:
            processed_results.append(result)
    
    logger.info(f"배치 처리 완료: {len(s3_urls)}개 이미지 중 {sum(1 for r in processed_results if r.get('success', False))}개 성공")
    return processed_results


def validate_and_prepare_s3_url(s3_url: str) -> str:
    """S3 URL 검증 및 전처리
    
    Args:
        s3_url: 검증할 S3 URL
        
    Returns:
        str: 검증된 S3 URL
        
    Raises:
        VisionS3URLError: URL이 유효하지 않은 경우
    """
    try:
        if not s3_url or not isinstance(s3_url, str):
            raise VisionS3URLError("유효하지 않은 S3 URL입니다.")
        
        s3_url = s3_url.strip()
        
        if not validate_s3_url_format(s3_url):
            raise VisionS3URLError(f"지원되지 않는 S3 URL 형식입니다: {s3_url}")
        
        return s3_url
        
    except S3InvalidURLError as e:
        raise VisionS3URLError(str(e))
    except Exception as e:
        raise VisionS3URLError(f"S3 URL 검증 중 오류: {str(e)}")


# 편의 함수들
async def quick_s3_ocr(s3_url: str) -> str:
    """간단한 S3 이미지 OCR - 텍스트만 반환
    
    Args:
        s3_url: S3 이미지 URL
        
    Returns:
        str: 추출된 텍스트
        
    Raises:
        VisionS3Error: 처리 실패 시
    """
    result = await process_s3_image_with_vision(s3_url)
    if result.get("success", False):
        return result.get("text", "")
    else:
        raise VisionS3Error(f"OCR 처리 실패: {result.get('error', 'Unknown error')}")


async def s3_ocr_with_paragraphs(s3_url: str) -> Dict[str, Any]:
    """S3 이미지 OCR - 텍스트와 문단 정보 반환
    
    Args:
        s3_url: S3 이미지 URL
        
    Returns:
        Dict[str, Any]: {"text": str, "paragraphs": List[str]}
        
    Raises:
        VisionS3Error: 처리 실패 시
    """
    result = await process_s3_image_with_vision(s3_url)
    if result.get("success", False):
        return {
            "text": result.get("text", ""),
            "paragraphs": result.get("paragraphs", [])
        }
    else:
        raise VisionS3Error(f"OCR 처리 실패: {result.get('error', 'Unknown error')}") 