import re
from typing import List

# 종결부호와 닫힘 따옴표
_SENT_END = r"[\.!\?…]+"
_RIGHT_QUOTE = r"[\"'”’]?"

# lookbehind 없이 캡처: 종결부호(+닫힘 따옴표)까지 포함하거나 입력 끝까지
_SENT_CAPTURE_RE = re.compile(rf"[^\.!\?…]+?(?:{_SENT_END}{_RIGHT_QUOTE}|$)")

def _normalize_spaces_keep_newline(text: str) -> str:
    # 개행은 유지, 공백만 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" \n+", "\n", text)
    text = re.sub(r"\n+ ", "\n", text)
    return text.strip()

def _tidy(sentence: str) -> str:
    # 중복 공백, 구두점 앞 공백 제거
    s = re.sub(r"\s+", " ", sentence)
    s = re.sub(r"\s+([,\.!\?…])", r"\1", s)
    return s.strip()

def split_sentences(text: str, split_on_newline: bool = False) -> List[str]:
    """
    텍스트 → 문장 리스트
    - 기본: 종결부호(. ! ? …) 기준
    - split_on_newline=True: 종결부호가 없을 땐 \n도 문장 경계로 보조 분리
    """
    if not text:
        return []
    t = _normalize_spaces_keep_newline(text)


    # 문장 캡처
    raw_parts = _SENT_CAPTURE_RE.findall(t)
    parts: List[str] = []
    for p in raw_parts:
        p = p.strip()
        if not p:
            continue
        if split_on_newline and ("\n" in p) and not re.search(_SENT_END, p):
            # 종결부호 없는 조각은 개행으로 보조 분리
            for seg in re.split(r"\n+", p):
                seg = seg.strip()
                if seg:
                    parts.append(_tidy(seg))
        else:
            parts.append(_tidy(p))
    return parts
