import os
import base64
from typing import Any, Dict, List

import httpx


# --- 내부 유틸: 시크릿/환경변수에서 키 읽기 (Docker secrets 지원) ---
def _read_secret(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return None


def _get_api_key() -> str:
    key = os.getenv("VISION_API_KEY")
    if not key:
        raise RuntimeError("환경변수 VISION_API_KEY가 필요합니다.")
    return key

def _get_api_url() -> str:
    return f"https://vision.googleapis.com/v1/images:annotate?key={_get_api_key()}"

# --- 공개 함수들 ---
def encode_bytes_to_b64(content: bytes) -> str:
    return base64.b64encode(content).decode("utf-8")


def build_payload(
    image_b64: str,
    feature: str = "TEXT_DETECTION",  # 문서 OCR이면 "DOCUMENT_TEXT_DETECTION" 권장
    language_hints: List[str] | None = None,
) -> Dict[str, Any]:
    req: Dict[str, Any] = {
        "image": {"content": image_b64},
        "features": [{"type": feature}],
    }
    if language_hints:
        req["imageContext"] = {"languageHints": language_hints}
    return {"requests": [req]}


async def call_vision_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = _get_api_url()
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(url, json=payload)
        res.raise_for_status()
        return res.json()


def extract_full_text(resp_json: Dict[str, Any]) -> str:
    resp = resp_json.get("responses", [{}])[0]
    text = resp.get("fullTextAnnotation", {}).get("text")
    if text:
        return text.strip()
    ann = resp.get("textAnnotations", [])
    if ann:
        return ann[0].get("description", "").strip()
    return ""


def extract_word_boxes(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    textAnnotations[1:] 의 단어 단위 바운딩 박스를 반환
    [{'text': '...', 'vertices': [{'x':..,'y':..}, ...]}, ...]
    """
    out: List[Dict[str, Any]] = []
    resp = resp_json.get("responses", [{}])[0]
    for i, item in enumerate(resp.get("textAnnotations", [])):
        if i == 0:
            continue  # [0]은 전체 문장
        out.append(
            {
                "text": item.get("description", ""),
                "vertices": item.get("boundingPoly", {}).get("vertices", []),
            }
        )
    return out
