import os, json
from typing import List, Dict, Any, Optional
from openai import OpenAI

def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 필요합니다.")
    return OpenAI(api_key=api_key)

def evaluate_segmentation_with_openai(
    text: str,
    sentences: List[str],
    model: str = "gpt-4.1-mini",
) -> Dict[str, Any]:
    """
    원문 text와 분리된 sentences를 넘기면,
    - 점수(0~1), 판단(OK/REVISE), 문제 목록, 추천 수정, 수정된 문장 배열을 JSON으로 반환
    """
    client = _client()
    sys = (
        "You are a careful Korean text segmentation evaluator. "
        "Assess whether the given list of sentences is a correct segmentation "
        "of the original text. Respond in JSON with keys: "
        "`score` (0~1 float), `verdict` ('OK'|'REVISE'), "
        "`issues` (string[]), `suggestions` (string[]), `fixed_sentences` (string[]). "
        "Keep `fixed_sentences` length close to the original sentences length; merge/split as needed."
    )
    user = {
        "original_text": text,
        "sentences": sentences,
    }
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        temperature=0.1,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except Exception:
        data = {"score": 0.0, "verdict": "REVISE", "issues": ["JSON parse error"], "suggestions": [], "fixed_sentences": []}
    data["model"] = model
    data["raw"] = content
    return data
