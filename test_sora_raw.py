import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()


async def test_sora_raw():
    api_key = os.getenv("OPENAI_API_KEY")
    url = "https://api.openai.com/v1/videos/generations"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    payload = {
        "model": "sora-2",
        "prompt": "Test video of a cat walking in a garden. [dialogue] cat: Meow.",
        "size": "720x1280",
        "seconds": "4",
    }

    print("Requesting Sora...")
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        print(f"Status Code: {response.status_code}")
        print("Response Body:")
        print(response.text)


if __name__ == "__main__":
    asyncio.run(test_sora_raw())
