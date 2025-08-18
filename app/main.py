import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

from .schemas.ocr import OCRRequestParams, OCRResponse
from .services.vision_ocr import (
    encode_bytes_to_b64,
    build_payload,
    call_vision_api,
    extract_full_text,
    extract_word_boxes,
)



app = FastAPI(title="Vision OCR Wrapper", version="1.0.0")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr", response_model=OCRResponse)
async def ocr_image(
    file: UploadFile = File(..., description="이미지 파일"),
    feature: str = Form("TEXT_DETECTION"),
    language_hints: str = Form("ko,en"),
    include_boxes: bool = Form(False),
    debug: bool = Form(False),
):
    """
    컨트롤러는 파라미터/유효성/에러 처리만.
    비즈니스 로직은 services/vision_ocr.py에서 수행.
    """
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="빈 파일입니다.")
        if len(content) > 6 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="파일이 너무 큽니다(>6MB).")

        image_b64 = encode_bytes_to_b64(content)
        langs = [s.strip() for s in (language_hints or "").split(",") if s.strip()]

        payload = build_payload(image_b64, feature=feature, language_hints=langs)
        vision_raw = await call_vision_api(payload)

        text = extract_full_text(vision_raw)
        result = {"text": text}

        if include_boxes:
            result["boxes"] = extract_word_boxes(vision_raw)
        if debug:
            result["raw"] = vision_raw

        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")
