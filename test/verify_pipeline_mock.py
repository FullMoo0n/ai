
import asyncio
import sys
import os

# Add app directory to path
sys.path.append(os.getcwd())

from app.services.integrated_pipeline import IntegratedPipeline
from app.services.sync_pipeline import SyncIntegratedPipeline

async def test_async_pipeline_mock():
    print("Testing Async IntegratedPipeline Mock...")
    pipeline = IntegratedPipeline("test_task_async")
    # Using a dummy Azure Blob URL
    result = await pipeline.execute("https://stdevbinary.blob.core.windows.net/blob-binary/test_image.jpg")
    
    assert result['status'] == 'completed'
    assert result['video_urls'][0] == "https://stdevbinary.blob.core.windows.net/blob-binary/KSL_Video_Generation_Request.mp4"
    assert "Mocked Text" in result['video_details'][0]['full_text']
    print("✅ Async Pipeline Mock Verified")

def test_sync_pipeline_mock():
    print("Testing Sync IntegratedPipeline Mock...")
    pipeline = SyncIntegratedPipeline("test_task_sync")
    # Using a dummy Azure Blob URL
    result = pipeline.execute("https://stdevbinary.blob.core.windows.net/blob-binary/test_image.jpg")
    
    assert result['status'] == 'completed'
    assert result['video_urls'][0] == "https://stdevbinary.blob.core.windows.net/blob-binary/KSL_Video_Generation_Request.mp4"
    assert "Mocked Text" in result['text']
    print("✅ Sync Pipeline Mock Verified")

if __name__ == "__main__":
    try:
        asyncio.run(test_async_pipeline_mock())
        test_sync_pipeline_mock()
        print("\n🎉 All Pipeline Mock Verification Tests Passed!")
    except Exception as e:
        print(f"\n❌ Verification Failed: {e}")
        sys.exit(1)
