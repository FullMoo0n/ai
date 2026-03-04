import os
import logging
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from app.services.azure_service import (
    get_azure_public_url,
    check_azure_blob_exists,
)

logger = logging.getLogger(__name__)


class OpenAIVisionError(Exception):
    """OpenAI Vision 처리 관련 에러"""

    pass


async def process_s3_image_with_vision(
    s3_url: str,
    feature: str = "TEXT_DETECTION",
    language_hints: Optional[List[str]] = None,
    include_word_boxes: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    이미지 URL을 OpenAI (gpt-4o)에 전달하여 OCR(텍스트 추출)을 수행합니다.
    기존 Google Vision API 인터페이스(vision_s3.py)와의 호환성을 유지하기 위해 같은 파라미터를 받습니다.

    Args:
        s3_url: 처리할 이미지 URL (Azure Blob URL 또는 일반 웹 URL)
        feature: (호환성 유지용)
        language_hints: (호환성 유지용)
        include_word_boxes: (호환성 유지용 - OpenAI는 단어 박스를 제공하지 않으므로 무시됨)

    Returns:
        Dict: OCR 처리 결과
            - success: 처리 성공 여부
            - text: OCR 추출 텍스트
            - image_context: 이미지 시각 맥락 요약(인물/배경/조명)
            - public_url: 실제 Vision 호출에 사용된 URL
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIVisionError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")

        client = AsyncOpenAI(api_key=api_key)

        # URL 처리 - Azure Blob Storage인 경우와 일반 URL인 경우 처리
        public_url = s3_url
        if "blob.core.windows.net" in s3_url:
            logger.info(f"Azure 이미지 OCR 처리 시작: {s3_url}")
            if not check_azure_blob_exists(s3_url):
                logger.warning(
                    f"Azure Blob을 찾을 수 없거나 접근할 수 없습니다: {s3_url}"
                )

            # 읽기 전용 SAS 토큰이 포함된 URL 생성 시도 (만약 public 컨테이너가 아니라면 필요함)
            try:
                public_url = get_azure_public_url(s3_url, expires_in_hours=1)
            except Exception as e:
                logger.warning(
                    f"SAS URL 발급 실패(public 컨테이너로 가정하고 원본 URL 사용): {e}"
                )
        else:
            logger.info(f"일반 웹 이미지 URL 감지: {s3_url}")

        prompt = "Extract all the text from this image exactly as it appears. Ensure the line breaks and spacing reflect the original structure. Only return the text, no other descriptions."
        if language_hints and "ko" in language_hints:
            prompt = "이 이미지에 있는 모든 텍스트를 있는 그대로 정확하게 추출해주세요. 원본 구조에 맞게 줄바꿈과 띄어쓰기를 유지하고, 설명 없이 텍스트만 출력해주세요."

        logger.info("OpenAI GPT-4o Vision API 호출 시작...")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": public_url, "detail": "high"},
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )

        extracted_text = response.choices[0].message.content.strip()

        # 마크다운 코드블록 제거 로직 (가끔 LLM이 ```text ... ``` 형태로 줄 때가 있음)
        if extracted_text.startswith("```"):
            lines = extracted_text.split("\n")
            if len(lines) >= 2:
                # 첫 줄과 마지막 줄 제거
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                extracted_text = "\n".join(lines).strip()

        image_context = ""
        try:
            context_prompt = (
                "이 이미지를 기반으로 영상 생성에 바로 사용할 수 있도록, "
                "다음 요소만 1~3문장으로 요약해줘: "
                "(1) 주체/인물 외형, (2) 배경/공간, (3) 조명/색감. "
                "텍스트를 읽거나 번역하지 말고 시각 정보만 설명해."
            )
            context_response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": context_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": public_url, "detail": "high"},
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            image_context = (context_response.choices[0].message.content or "").strip()
            logger.info("OpenAI 이미지 맥락 요약 완료")
        except Exception as context_error:
            logger.warning(f"이미지 맥락 요약 실패 (OCR은 계속 진행): {context_error}")

        result = {
            "success": True,
            "text": extracted_text,
            "image_context": image_context,
            "s3_url": s3_url,  # (호환성 유지용 키)
            "image_url": s3_url,
            "public_url": public_url,
        }

        logger.info(f"OpenAI OCR 처리 완료: {len(extracted_text)} 글자 추출")
        return result

    except Exception as e:
        logger.error(f"OpenAI 이미지 OCR 처리 중 오류: {str(e)}")
        raise OpenAIVisionError(f"OpenAI OCR 이미지 처리 실패: {str(e)}")
