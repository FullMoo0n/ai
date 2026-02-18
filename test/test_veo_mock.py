
import pytest
import asyncio
from app.services.veo_service import VeoService

@pytest.mark.asyncio
async def test_veo_service_mock_generation():
    """Veo 서비스가 배포 모드에서 Mock 응답을 반환하는지 테스트"""
    service = VeoService(api_key="test_key")
    
    prompt = "Test prompt for sign language video"
    task_id = "test_task_123"
    
    # 실제 API 호출 없이 Mock 응답이 즉시 반환되어야 함
    result = await service.generate_sign_video(
        prompt=prompt,
        task_id=task_id
    )
    
    # 검증
    assert result['status'] == 'success'
    assert result['video_url'] == 'https://stdevbinary.blob.core.windows.net/blob-binary/KSL_Video_Generation_Request.mp4'
    assert result['prompt'] == prompt
    assert result['task_id'] == task_id
    assert 'note' in result
    assert 'Mock' in result['note']
    
    print("✅ Veo Service Mock 동작 확인 완료")

if __name__ == "__main__":
    asyncio.run(test_veo_service_mock_generation())
