from typing import List, Optional, Any
from pydantic import BaseModel, Field




class BoxedText(BaseModel):
    text: str
    box: List[int] = Field(..., description="[x1,y1,x2,y2]")


class OCRResponse(BaseModel):
    text: str
    paragraphs: Optional[List[BoxedText]] = None
    word_boxes: Optional[List[BoxedText]] = None
    raw: Optional[dict] = None



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
