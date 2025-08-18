from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class ValidateRequest(BaseModel):
    text: str
    sentences: List[str]

class ValidateResponse(BaseModel):
    score: float
    verdict: str
    issues: List[str]
    suggestions: List[str]
    fixed_sentences: List[str]
    raw: str
