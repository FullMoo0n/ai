import re
from typing import List

# 한글/영문/숫자는 묶고, 그 외 기호는 단독 토큰
_TOKEN_RE = re.compile(r"[가-힣]+|[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9가-힣]")

def tokenize(sentence: str) -> List[str]:
    if not sentence:
        return []
    return [m.group(0) for m in _TOKEN_RE.finditer(sentence)]
