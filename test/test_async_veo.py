#!/usr/bin/env python3
"""
비동기 Veo 비디오 생성 API 테스트 스크립트
"""

import requests
import time
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_async_veo_generation():
    """비동기 Veo 비디오 생성 테스트"""
    
    # 1. 비동기 비디오 생성 요청
    print("=== 비동기 비디오 생성 테스트 ===")
    print("1. 비디오 생성 요청 중...")
    
    payload = {
        "prompt": "A red panda riding a skateboard in a sunny park with trees and blue sky",
        "aspect_ratio": "16:9",
        "timeout_seconds": 600
    }
    
    response = requests.post(f"{BASE_URL}/veo/async", json=payload)
    
    if response.status_code != 202:
        print(f"❌ 요청 실패: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    task_id = result["task_id"]
    print(f"✅ 작업이 큐에 추가됨")
    print(f"   Task ID: {task_id}")
    print(f"   상태: {result['status']}")
    print(f"   메시지: {result['message']}")
    print()
    
    # 2. 작업 상태 주기적 조회
    print("2. 작업 상태 모니터링...")
    print("   (Ctrl+C로 중단 가능)")
    print()
    
    start_time = time.time()
    
    while True:
        try:
            # 상태 조회
            status_response = requests.get(f"{BASE_URL}/veo/status/{task_id}")
            
            if status_response.status_code != 200:
                print(f"❌ 상태 조회 실패: {status_response.status_code}")
                break
            
            status = status_response.json()
            elapsed = int(time.time() - start_time)
            
            print(f"[{elapsed:3d}초] 상태: {status['status']}")
            
            if status["status"] == "PENDING":
                print("   대기 중...")
            
            elif status["status"] == "PROGRESS":
                progress = status.get("progress", {})
                print(f"   진행 상황: {progress.get('message', 'N/A')}")
                if "elapsed_seconds" in progress:
                    print(f"   작업 경과: {progress['elapsed_seconds']}초")
            
            elif status["status"] == "SUCCESS":
                result = status.get("result", {})
                print("✅ 작업 완료!")
                print(f"   상태: {result.get('status', 'N/A')}")
                print(f"   메시지: {result.get('message', 'N/A')}")
                if result.get("video_uri"):
                    print(f"   비디오 URI: {result['video_uri']}")
                else:
                    print("   ⚠️ 비디오 URI가 없습니다")
                break
            
            elif status["status"] == "FAILURE":
                print("❌ 작업 실패!")
                print(f"   오류: {status.get('error', 'Unknown error')}")
                break
            
            else:
                print(f"   알 수 없는 상태: {status['status']}")
            
            print()
            time.sleep(10)  # 10초마다 확인
            
        except KeyboardInterrupt:
            print("\n사용자에 의해 중단됨")
            
            # 작업 취소 여부 확인
            cancel = input("작업을 취소하시겠습니까? (y/N): ").strip().lower()
            if cancel in ['y', 'yes']:
                try:
                    cancel_response = requests.delete(f"{BASE_URL}/veo/cancel/{task_id}")
                    if cancel_response.status_code == 200:
                        print("✅ 작업이 취소되었습니다")
                    else:
                        print(f"❌ 작업 취소 실패: {cancel_response.status_code}")
                except Exception as e:
                    print(f"❌ 작업 취소 중 오류: {e}")
            break
        
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            break


def test_sync_veo_generation():
    """동기 Veo 비디오 생성 테스트 (짧은 타임아웃)"""
    
    print("=== 동기 비디오 생성 테스트 (짧은 타임아웃) ===")
    
    payload = {
        "prompt": "A simple animation of a bouncing ball",
        "aspect_ratio": "16:9", 
        "timeout_seconds": 60  # 1분으로 제한하여 타임아웃 테스트
    }
    
    print("비디오 생성 요청 중... (60초 타임아웃)")
    
    try:
        response = requests.post(f"{BASE_URL}/veo", json=payload, timeout=70)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 동기 생성 완료")
            print(f"   상태: {result.get('status', 'N/A')}")
            print(f"   메시지: {result.get('message', 'N/A')}")
            if result.get("video_uri"):
                print(f"   비디오 URI: {result['video_uri']}")
        else:
            print(f"❌ 요청 실패: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.Timeout:
        print("⏰ 요청 타임아웃 (예상된 결과)")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def check_service_health():
    """서비스 상태 확인"""
    
    print("=== 서비스 상태 확인 ===")
    
    try:
        # Health check
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI 서버 정상")
        else:
            print(f"❌ FastAPI 서버 오류: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FastAPI 서버 연결 실패: {e}")
        return False
    
    # API 문서 접근 확인
    try:
        response = requests.get(f"{BASE_URL}/docs", timeout=5)
        if response.status_code == 200:
            print("✅ API 문서 접근 가능")
        else:
            print(f"⚠️ API 문서 접근 문제: {response.status_code}")
    except Exception as e:
        print(f"⚠️ API 문서 접근 실패: {e}")
    
    print()
    return True


if __name__ == "__main__":
    print("Veo 비동기 비디오 생성 API 테스트")
    print("=" * 50)
    print()
    
    # 서비스 상태 확인
    if not check_service_health():
        print("서비스가 실행 중인지 확인하세요:")
        print("1. uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")
        print("2. ./scripts/start_celery.sh")
        print("3. docker-compose up -d redis")
        exit(1)
    
    # 테스트 선택
    print("테스트 선택:")
    print("1. 비동기 비디오 생성 테스트 (권장)")
    print("2. 동기 비디오 생성 테스트 (타임아웃 테스트)")
    print("3. 둘 다 실행")
    
    choice = input("\n선택 (1-3): ").strip()
    print()
    
    if choice == "1":
        test_async_veo_generation()
    elif choice == "2":
        test_sync_veo_generation()
    elif choice == "3":
        test_async_veo_generation()
        print("\n" + "=" * 50 + "\n")
        test_sync_veo_generation()
    else:
        print("잘못된 선택입니다.")
    
    print("\n테스트 완료!") 