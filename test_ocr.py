import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from app.services.vision_s3 import process_s3_image_with_vision


async def test_ocr():
    url = "https://stdevbinary.blob.core.windows.net/blob-binary/abbbe9f6-7645-4b3e-bde9-c33dec4bde51.jpeg"
    print(f"Testing OCR with URL: {url}")
    try:
        result = await process_s3_image_with_vision(url)
        print("Success!")
        print(f"Extracted Text: {result.get('text', '')[:100]}...")
        print(f"Paragraphs found: {len(result.get('paragraphs', []))}")
    except Exception as e:
        import httpx

        if isinstance(e.args[0], str) and "400" in e.args[0]:
            print(f"Error occurred: {e}")
        elif hasattr(e, "__dict__"):
            print(f"Error occurred: {e}")
            if hasattr(e, "response") and e.response:
                print(e.response.text)
        else:
            print(f"Error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(test_ocr())
