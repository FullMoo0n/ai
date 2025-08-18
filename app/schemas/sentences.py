from typing import List, Optional
from pydantic import BaseModel

class SentencesRequest(BaseModel):
    # 하나만 써도 되고, 둘 다 쓰면 paragraphs를 우선 사용
    text: Optional[str] = None

class SentencesResponse(BaseModel):
    # 입력 전체를 하나로 합친 원문(요약용)
    text: str
    # 문장 리스트
    sentences: List[str]
