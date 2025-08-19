from typing import Optional, Any, Dict
from pydantic import BaseModel, Field
import time

try:
    from google import genai
    from google.genai.types import GenerateVideosConfig
    _has_google_genai = True
except Exception:
    _has_google_genai = False


class VeoRequest(BaseModel):
    """비디오 생성 요청 모델"""
    prompt: str = Field(
        ...,
        description="비디오 생성을 위한 텍스트 설명",
        min_length=10,
        max_length=500
    )
    aspect_ratio: Optional[str] = Field(
        "16:9",
        description="비디오 화면 비율"
    )
    model: Optional[str] = Field(
        "veo-2.0-generate-001",
        description="사용할 Veo 모델 버전"
    )
    timeout_seconds: Optional[int] = Field(
        600,  # 기본값을 10분으로 증가
        description="비디오 생성 대기 시간 (초)",
        ge=30,
        le=1200  # 최대 20분
    )

    class Config:
        schema_extra = {
            "example": {
                "prompt": "A majestic eagle soaring over snow-capped mountains at sunset",
                "aspect_ratio": "16:9",
                "model": "veo-2.0-generate-001",
                "timeout_seconds": 600
            }
        }


class VeoAsyncRequest(BaseModel):
    """비동기 비디오 생성 요청 모델"""
    prompt: str = Field(
        ...,
        description="비디오 생성을 위한 텍스트 설명",
        min_length=10,
        max_length=500
    )
    aspect_ratio: Optional[str] = Field(
        "16:9",
        description="비디오 화면 비율"
    )
    model: Optional[str] = Field(
        "veo-2.0-generate-001",
        description="사용할 Veo 모델 버전"
    )
    timeout_seconds: Optional[int] = Field(
        600,
        description="비디오 생성 대기 시간 (초)",
        ge=30,
        le=1200
    )

    class Config:
        schema_extra = {
            "example": {
                "prompt": "A red panda riding a skateboard in a sunny park",
                "aspect_ratio": "16:9",
                "model": "veo-2.0-generate-001",
                "timeout_seconds": 600
            }
        }


class VeoResponse(BaseModel):
    """비디오 생성 응답 모델"""
    status: str = Field(description="생성 상태 (completed, pending, completed_with_error, error)")
    video_uri: Optional[str] = Field(None, description="생성된 비디오 다운로드 URI")
    local_path: Optional[str] = Field(None, description="로컬에 저장된 비디오 파일 경로")
    operation: Optional[str] = Field(None, description="Google API Operation ID")
    message: Optional[str] = Field(None, description="상세 메시지")
    error: Optional[str] = Field(None, description="오류 메시지 (있는 경우)")


class VeoAsyncResponse(BaseModel):
    """비동기 비디오 생성 응답 모델"""
    task_id: str = Field(description="비동기 작업 ID")
    status: str = Field(description="작업 상태")
    message: str = Field(description="상태 메시지")


class VeoTaskStatus(BaseModel):
    """비동기 작업 상태 조회 응답 모델"""
    task_id: str = Field(description="작업 ID")
    status: str = Field(description="작업 상태 (PENDING, PROGRESS, SUCCESS, FAILURE, RETRY, REVOKED)")
    result: Optional[Dict[str, Any]] = Field(None, description="작업 결과 (완료시)")
    progress: Optional[Dict[str, Any]] = Field(None, description="진행 상황 정보")
    error: Optional[str] = Field(None, description="오류 메시지")


class ErrorResponse(BaseModel):
    """오류 응답 모델"""
    error: str = Field(description="오류 메시지")
    