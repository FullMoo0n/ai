import asyncio
from datetime import datetime
import uuid

from app.services.sora_service import generate_sign_video

PROMPT = """Generate a Korean Sign Language (KSL) video scene.

Character:
- A bear-like creature, standing upright with dark gray fur, no clothing, yellow eyes, and a confident expression, arms crossed.
- Keep the same illustrated character design consistently throughout the video.

Scene:
- (1) 주체/인물 외형: 커다란 회색 곰이 눈을 가늘게 뜨고 두 팔을 교차한 채 서 있다. 곰은 몸 전체에 작은 털이 나 있으며, 밝은 노란색 눈이 돋보인다.
(2) 배경/공간: 흰색 배경이고 곰 옆에 작은 부분의 호랑이 줄무늬가 보이며, 이는 곰과 다른 캐릭터들이 함께 있는 장면임을 암시한다.
(3) 조명/색감: 조명은 부드럽고 색감은 주로 중성적인 회색과 흰색이 사용되어 있으며, 노란색 눈이 강조된다.
- Match the original storybook illustration style and mood.

Signing content:
- Original text: 자, 이제 내 키가 더 크지?
- KSL gloss sequence (strict order): 이제 키가 크지
- Sign references:
- 키가: 왼손의 손바닥에 오른 주먹의 1지를 펴서 끝을 댄다.

Performance rules:
- Use natural KSL facial expressions and body language.
- Signing speed: slow and clear for children.
- Camera: fixed front-facing, upper body visible, both hands always in frame."""

REFERENCE_IMAGE_URL = "https://stdevbinary.blob.core.windows.net/blob-binary/IMG_3510.jpeg"
TASK_ID = f"sora_try_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"


async def main():
    result = await generate_sign_video(
        prompt=PROMPT,
        task_id=TASK_ID,
        reference_image_url=REFERENCE_IMAGE_URL,
    )

    print("=== SORA TRY RESULT ===")
    print("TASK_ID:", TASK_ID)
    print("STATUS:", result.get("status"))
    print("VIDEO_URL:", result.get("video_url"))
    print("NOTE:", result.get("note"))
    print("ERROR:", result.get("error"))
    print("REFERENCE_IMAGE_USED:", result.get("reference_image_used"))
    print("VIDEO_OUTPUT:", result.get("video_output"))
    print("REQUEST_METADATA:", result.get("request_metadata"))


if __name__ == "__main__":
    asyncio.run(main())
