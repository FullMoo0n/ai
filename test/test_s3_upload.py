#!/usr/bin/env python3
"""
S3 업로드 기능 테스트 스크립트
"""
import os
import sys
import time
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from app.tasks.veo_tasks import upload_video_to_s3


def test_s3_upload():
    """S3 업로드 기능을 테스트합니다."""
    
    print("🧪 S3 업로드 기능 테스트 시작...")
    print("=" * 60)
    
    # 테스트 URL
    test_video_uri = "https://generativelanguage.googleapis.com/download/v1beta/files/3pf5wmwdeq23:download?alt=media"
    test_task_id = f"test_{int(time.time())}"
    
    print(f"📹 테스트 비디오 URI: {test_video_uri}")
    print(f"🏷️  테스트 Task ID: {test_task_id}")
    print()
    
    # 환경변수 확인
    print("🔧 환경변수 확인:")
    required_env_vars = [
        "GOOGLE_API_KEY",
        "AWS_ACCESS_KEY_ID", 
        "AWS_SECRET_ACCESS_KEY",
        "AWS_REGION",
        "S3_BUCKET_NAME"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        value = os.getenv(var)
        if value:
            if var in ["GOOGLE_API_KEY", "AWS_SECRET_ACCESS_KEY"]:
                # 민감한 정보는 일부만 표시
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                print(f"  ✅ {var}: {masked_value}")
            else:
                print(f"  ✅ {var}: {value}")
        else:
            print(f"  ❌ {var}: 설정되지 않음")
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\n⚠️  누락된 환경변수: {', '.join(missing_vars)}")
        print("테스트를 계속 진행하지만 실패할 수 있습니다.")
    
    print("\n" + "=" * 60)
    print("🚀 S3 업로드 시작...")
    
    try:
        # S3 업로드 실행
        start_time = time.time()
        result = upload_video_to_s3(test_video_uri, test_task_id)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        print(f"⏱️  소요 시간: {elapsed_time:.2f}초")
        print(f"📄 결과: {result}")
        
        if "S3 업로드 완료" in result:
            print("\n✅ 테스트 성공! S3 업로드가 완료되었습니다.")
        else:
            print("\n❌ 테스트 실패. 결과를 확인해주세요.")
            
    except Exception as e:
        print(f"\n💥 테스트 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("=" * 60)
    print("🧪 테스트 완료")


if __name__ == "__main__":
    test_s3_upload() 