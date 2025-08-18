from typing import List
from pydantic import BaseModel

class TokensRequest(BaseModel):
    sentences: List[str]  # 문장 분리는 이미 끝났다는 가정

class TokensResponse(BaseModel):
    tokens_per_sentence: List[List[str]]
