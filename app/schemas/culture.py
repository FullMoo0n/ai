from pydantic import BaseModel
from typing import Optional


class CultureRequest(BaseModel):
    keyword: str


class CultureResponse(BaseModel):
    keyword: str
    sign_description: Optional[str] = None
    error: Optional[str] = None 