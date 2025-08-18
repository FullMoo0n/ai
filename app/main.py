from typing import List
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from PIL import Image

from .schemas.ocr import OCRResponse
from .services.vision_ocr import (
    encode_bytes_to_b64,
    build_payload,
    call_vision_api,
    extract_full_text,
    extract_word_boxes,
    extract_paragraphs_spatial_proximity_advanced,
)

# .env 로드
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

app = FastAPI(title="Vision OCR Wrapper", version="1.4.0")


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
