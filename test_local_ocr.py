import asyncio
import os
import httpx
from dotenv import load_dotenv
from app.services.vision_ocr import build_payload, call_vision_api, encode_bytes_to_b64

load_dotenv()


async def test_local_ocr():
    # 빈 이미지라도 만들어서 테스트
    content = b"fakeimagecontentjusttotestbadrequest"
    image_b64 = encode_bytes_to_b64(content)

    payload = build_payload(image_b64, feature="TEXT_DETECTION", language_hints=["ko"])

    try:
        result = await call_vision_api(payload)
        print("Success!")
    except httpx.HTTPStatusError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Other Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_local_ocr())
