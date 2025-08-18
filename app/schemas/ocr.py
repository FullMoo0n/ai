from pydantic import BaseModel, Field
from typing import List, Optional, Any


class OCRRequestParams(BaseModel):
    feature: str = Field(default="TEXT_DETECTION", description="또는 TEXT_DETECTION")
    language_hints: Optional[List[str]] = Field(default=["ko", "en"])
    include_boxes: bool = False
    debug: bool = False


class WordBox(BaseModel):
    text: str
    vertices: list[dict]


class OCRResponse(BaseModel):
    text: str
    boxes: Optional[List[WordBox]] = None
    raw: Optional[Any] = None
