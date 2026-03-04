import asyncio
import os
import httpx
from dotenv import load_dotenv

load_dotenv()


async def test_vision_400():
    api_key = os.getenv("VISION_API_KEY")
    url = f"https://vision.googleapis.com/v1/images:annotate?key={api_key}"
    image_url = "https://stdevbinary.blob.core.windows.net/blob-binary/abbbe9f6-7645-4b3e-bde9-c33dec4bde51.jpeg"

    payload = {
        "requests": [
            {
                "image": {"source": {"imageUri": image_url}},
                "features": [{"type": "TEXT_DETECTION", "maxResults": 1000}],
                "imageContext": {"languageHints": ["ko"]},
            }
        ]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.text)


if __name__ == "__main__":
    asyncio.run(test_vision_400())
