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
from io import BytesIO

from openai import AsyncOpenAI
from PIL import Image
from app.services.azure_service import upload_stream_to_azure

logger = logging.getLogger(__name__)


SORA_VIDEO_OUTPUT_BLOCK = """Video output requirements:
- Smooth, continuous sign language transitions with no abrupt pauses between signs.
- No subtitles, no captions, and no on-screen text of any kind.
- Keep a consistent cartoon / illustration visual style, not photorealistic."""


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

        # 비동기 클라이언트 사용
        self.client = AsyncOpenAI(api_key=self.api_key)
        logger.info("🤖 OpenAI Sora 비동기 클라이언트 초기화 완료")

    async def generate_sign_video(
        self,
        prompt: str,
        task_id: Optional[str] = None,
        reference_image_url: Optional[str] = None,
        auto_upload_to_azure: bool = True,
    ) -> Dict[str, Any]:
        """
        프롬프트를 사용하여 OpenAI Sora로 비디오 생성

        Args:
            prompt: 수어 비디오 프롬프트
            task_id: 작업 ID (로깅용)
            reference_image_url: 원본 참조 이미지 URL (첫 프레임 캐릭터 고정용)
            auto_upload_to_azure: MP4 생성 후 Azure 자동 업로드 여부 (기본: True)

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
            full_prompt = self._compose_sora_prompt(prompt)
            logger.info(f"🎬 Sora 비디오 생성 시작: {task_id or 'unknown'}")
            logger.info(f"📝 프롬프트 길이: {len(full_prompt)}자")
            requested_size = "1280x720"
            requested_seconds = "4"

            request_metadata = {
                "task_id": task_id or "unknown",
                "source": "ksl-pipeline",
                "reference_image_url": reference_image_url or "",
            }

            input_reference = None
            reference_image_used = False
            if reference_image_url:
                try:
                    input_reference = await self._build_input_reference(
                        reference_image_url,
                        requested_size,
                    )
                    reference_image_used = True
                    logger.info("🖼️ 참조 이미지 연결 완료 (input_reference)")
                except Exception as ref_err:
                    logger.warning(
                        f"⚠️ 참조 이미지 연결 실패, 텍스트 프롬프트로 계속 진행: {ref_err}"
                    )

            # API 호출
            try:
                # sora-2 모델 사용, 비동기 폴링 (생성 시간 대기)
                request_payload = {
                    "model": "sora-2",
                    "prompt": full_prompt,
                    "input_reference": input_reference,
                    "seconds": requested_seconds,
                    "size": requested_size,
                    "metadata": request_metadata,
                }

                try:
                    response = await self.client.videos.create_and_poll(
                        **request_payload
                    )
                except TypeError as type_error:
                    if "metadata" in str(type_error):
                        logger.warning(
                            "⚠️ SDK에서 videos.create_and_poll(metadata=...)를 지원하지 않아 metadata 없이 재시도합니다."
                        )
                        request_payload.pop("metadata", None)
                        response = await self.client.videos.create_and_poll(
                            **request_payload
                        )
                    else:
                        raise
                logger.info(f"⏳ Sora 응답 완료: {response}")
            except Exception as e:
                logger.error(
                    f"⚠️ 현재 OpenAI Python SDK에서 Sora API 오류 발생. 예외: {str(e)}"
                )
                raise SoraServiceError(f"Sora API 호출 실패: {str(e)}")

            status = getattr(response, "status", None)
            if status != "completed":
                error_obj = getattr(response, "error", None)
                raise SoraServiceError(
                    f"Sora 비디오 생성 미완료(status={status}, error={error_obj})"
                )

            video_id = getattr(response, "id", None)
            if not video_id:
                raise SoraServiceError("Sora API 응답에 video id가 없습니다.")

            binary_content = await self.client.videos.download_content(video_id)
            video_bytes = binary_content.content
            content_type = binary_content.response.headers.get("content-type", "video/mp4")

            if len(video_bytes) < 1000:
                raise SoraServiceError(
                    f"다운로드된 비디오가 비정상적으로 작습니다 ({len(video_bytes)} bytes)"
                )

            azure_url: Optional[str] = None
            upload_status = "skipped"

            if auto_upload_to_azure:
                azure_url = await self._upload_video_bytes_to_azure(
                    video_bytes=video_bytes,
                    content_type=content_type,
                    task_id=task_id,
                )
                upload_status = "uploaded"
            else:
                logger.warning("⚠️ Azure 자동 업로드 비활성화(auto_upload_to_azure=False)")

            return {
                "status": "success",
                "video_url": azure_url,
                "azure_upload_status": upload_status,
                "prompt": full_prompt,
                "task_id": task_id,
                "reference_image_url": reference_image_url,
                "reference_image_used": reference_image_used,
                "request_metadata": request_metadata,
                "created_at": datetime.now().isoformat(),
                "note": "Sora API Video Generation",
            }

        except Exception as e:
            logger.error(f"❌ Sora 수어 비디오 생성 중 오류: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "prompt": full_prompt,
                "task_id": task_id,
                "azure_upload_status": "error",
                "reference_image_url": reference_image_url,
                "request_metadata": request_metadata,
                "created_at": datetime.now().isoformat(),
                "note": f"Sora API 오류: {str(e)}",
            }

    def _compose_sora_prompt(self, prompt: str) -> str:
        """Sora 공통 출력 규칙을 프롬프트 본문에 포함"""
        stripped_prompt = (prompt or "").strip()
        if not stripped_prompt:
            return SORA_VIDEO_OUTPUT_BLOCK

        if "Video output" in stripped_prompt or "Video output requirements" in stripped_prompt:
            return stripped_prompt

        return f"""{SORA_VIDEO_OUTPUT_BLOCK}

Scene description:
{stripped_prompt}"""

    async def _build_input_reference(self, reference_image_url: str, requested_size: str):
        """원본 이미지 URL을 Sora input_reference 형식으로 변환"""
        timeout = httpx.Timeout(20.0, connect=10.0)
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(reference_image_url)
            response.raise_for_status()

        image_bytes = response.content
        if not image_bytes:
            raise SoraServiceError("참조 이미지 바이트가 비어 있습니다.")

        width, height = self._parse_size(requested_size)
        resized_bytes = self._resize_image_to_match(image_bytes, width, height)

        content_type = "image/jpeg"
        filename = "reference_image.jpg"
        return (filename, resized_bytes, content_type)

    def _parse_size(self, size: str) -> tuple[int, int]:
        """'1280x720' 형식 문자열을 (width, height)로 파싱"""
        try:
            width_str, height_str = size.lower().split("x")
            return int(width_str), int(height_str)
        except Exception as e:
            raise SoraServiceError(f"잘못된 size 형식: {size} ({e})")

    def _resize_image_to_match(self, image_bytes: bytes, width: int, height: int) -> bytes:
        """참조 이미지를 지정 해상도로 리사이즈하여 JPEG 바이트로 반환"""
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                converted = image.convert("RGB")
                resized = converted.resize((width, height), Image.Resampling.LANCZOS)
                output = BytesIO()
                resized.save(output, format="JPEG", quality=95)
                return output.getvalue()
        except Exception as e:
            raise SoraServiceError(f"참조 이미지 리사이즈 실패: {e}")

    async def _upload_video_bytes_to_azure(
        self,
        video_bytes: bytes,
        content_type: str,
        task_id: Optional[str],
    ) -> str:
        """Upload downloaded video bytes to Azure Blob Storage"""
        try:
            # Generate blob name
            timestamp = int(time.time())
            blob_name = f"sora-videos/sora_video_{task_id or 'unknown'}_{timestamp}.mp4"

            # Upload to Azure
            azure_url = await upload_stream_to_azure(
                stream=video_bytes, blob_name=blob_name, content_type=content_type
            )

            return azure_url

        except Exception as e:
            logger.error(f"❌ 비디오 Azure 업로드 중 실패: {str(e)}")
            raise SoraServiceError(f"Azure 업로드 실패: {str(e)}")


# 편의 함수
async def generate_sign_video(
    prompt: str,
    task_id: Optional[str] = None,
    reference_image_url: Optional[str] = None,
    auto_upload_to_azure: bool = True,
) -> Dict[str, Any]:
    """
    편의 함수: 수어 비디오 생성 (Sora)
    """
    service = SoraService()
    return await service.generate_sign_video(
        prompt,
        task_id,
        reference_image_url,
        auto_upload_to_azure,
    )
