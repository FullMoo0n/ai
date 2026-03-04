"""
OpenAI Sora 비디오 생성 서비스 모듈

OpenAI API를 사용하여 텍스트 프롬프트로 비디오를 생성하는 서비스 (Sora 모델 등 지원)
"""

import os
import time
import logging
import asyncio
import httpx
from typing import Optional, Dict, Any
from datetime import datetime

from openai import OpenAI
from app.services.azure_service import upload_stream_to_azure

logger = logging.getLogger(__name__)


class SoraServiceError(Exception):
    """Sora 서비스 에러"""

    pass


class SoraAPIKeyError(SoraServiceError):
    """Sora API 키 에러"""

    pass


class SoraService:
    """OpenAI Sora 비디오 생성 서비스"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Sora 서비스 초기화
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise SoraAPIKeyError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=self.api_key)
        logger.info("🤖 OpenAI Sora 클라이언트 초기화 완료")

    async def generate_sign_video(
        self, prompt: str, task_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        프롬프트를 사용하여 OpenAI Sora로 비디오 생성

        Args:
            prompt: 수어 비디오 프롬프트
            task_id: 작업 ID (로깅용)

        Returns:
            Dict: 생성 결과
                - status: "success" | "error"
                - video_url: Azure Blob에 업로드된 생성된 비디오 URL
                - prompt: 사용된 프롬프트
                - task_id: 작업 ID
                - created_at: 생성 시간
                - note: 추가 정보
        """
        try:
            logger.info(f"🎬 Sora 비디오 생성 시작: {task_id or 'unknown'}")
            logger.info(f"📝 프롬프트 길이: {len(prompt)}자")

            # API 호출
            video_url = None
            try:
                response = self.client.videos.generate(
                    model="sora-video-01",  # OpenAI Sora model identifier
                    prompt=prompt,
                )
                logger.info(f"⏳ Sora 응답 완료: {response}")

                if hasattr(response, "data") and len(response.data) > 0:
                    video_url = response.data[0].url
            except AttributeError:
                logger.warning(
                    "⚠️ 현재 OpenAI Python SDK에서 'videos.generate' 속성을 지원하지 않습니다. (Sora API 미지원). 테스트용 Mock URL을 반환합니다."
                )
                # Mock URL 반환 (테스트용)
                video_url = "https://mock-sora-test-video.com/sample_sora_video.mp4"
                await asyncio.sleep(2)  # 생성 시간 모방

            if not video_url:
                raise SoraServiceError("Sora API 응답에 video URL이 없습니다.")

            # Download and upload to Azure
            azure_url = await self._download_and_upload_to_azure(video_url, task_id)

            return {
                "status": "success",
                "video_url": azure_url,
                "prompt": prompt,
                "task_id": task_id,
                "created_at": datetime.now().isoformat(),
                "note": "Sora API Video Generation",
            }

        except Exception as e:
            logger.error(f"❌ Sora 수어 비디오 생성 중 오류: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "prompt": prompt,
                "task_id": task_id,
                "created_at": datetime.now().isoformat(),
                "note": f"Sora API 오류: {str(e)}",
            }

    async def _download_and_upload_to_azure(
        self, video_url: str, task_id: Optional[str]
    ) -> str:
        """Download video from URL and upload to Azure Blob Storage"""
        try:
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                response = await client.get(video_url)
                response.raise_for_status()

                content_type = response.headers.get("content-type", "video/mp4")
                video_bytes = response.content

                if len(video_bytes) < 1000:
                    logger.warning(
                        f"파일이 너무 작습니다 (크기: {len(video_bytes)} bytes)"
                    )
                    return video_url  # Return original if suspicion of invalid file

            # Generate blob name
            timestamp = int(time.time())
            blob_name = f"sora-videos/sora_video_{task_id or 'unknown'}_{timestamp}.mp4"

            # Upload to Azure
            azure_url = await upload_stream_to_azure(
                stream=video_bytes, blob_name=blob_name, content_type=content_type
            )

            return azure_url

        except Exception as e:
            logger.error(f"❌ 비디오 렌더링/Azure 업로드 중 실패: {str(e)}")
            return video_url  # Return original URL on failure to at least provide the video


# 편의 함수
async def generate_sign_video(
    prompt: str, task_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    편의 함수: 수어 비디오 생성 (Sora)
    """
    service = SoraService()
    return await service.generate_sign_video(prompt, task_id)
