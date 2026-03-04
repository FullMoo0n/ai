import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv()


async def try_endpoints():
    api_key = os.getenv("OPENAI_API_KEY")
    endpoints = [
        "https://api.openai.com/v1/video/generations",
        "https://api.openai.com/v1/videos/generate",
        "https://api.openai.com/v1/video/generate",
        "https://api.openai.com/v1/generations/video",
    ]

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "sora-2",
        "prompt": "Test video",
        "size": "720x1280",
        "seconds": "4",
    }

    async with httpx.AsyncClient() as client:
        for url in endpoints:
            print(f"Trying: {url}")
            response = await client.post(url, headers=headers, json=payload)
            print(f"Status: {response.status_code}")
            if response.status_code != 404 and response.status_code != 405:
                print(response.text)


if __name__ == "__main__":
    asyncio.run(try_endpoints())
