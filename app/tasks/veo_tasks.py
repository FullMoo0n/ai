import os
import time
import asyncio
import aiofiles
from pathlib import Path
from typing import Dict, Any
from celery import current_task
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import io

from ..celery_app import celery_app

# Google GenAI SDK 사용
try:
    from google import genai
    from google.genai.types import GenerateVideosConfig
    import httpx
    _has_genai = True
except ImportError:
    _has_genai = False


def upload_video_to_s3(video_uri: str, task_id: str) -> str:
    """
    Google API에서 비디오를 다운로드하여 S3에 업로드합니다.
    """
    try:
        # Google API Key 가져오기
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            return "업로드 실패: GOOGLE_API_KEY 환경변수가 설정되지 않았습니다."
        
        # AWS 설정 확인
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "ap-northeast-2")
        s3_bucket = os.getenv("S3_BUCKET_NAME")
        
        if not all([aws_access_key, aws_secret_key, s3_bucket]):
            return "업로드 실패: AWS 설정이 완료되지 않았습니다. (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME 필요)"
        
        # S3 클라이언트 초기화
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # HTTP 클라이언트로 비디오 다운로드 (API 키를 쿼리 파라미터로)
        if "?" in video_uri:
            download_url = f"{video_uri}&key={google_api_key}"
        else:
            download_url = f"{video_uri}?key={google_api_key}"
        
        with httpx.Client(timeout=300, follow_redirects=True) as client:
            response = client.get(download_url)
            response.raise_for_status()
            
            # Content-Type 확인
            content_type = response.headers.get('content-type', 'video/mp4')
            if len(response.content) < 1000:
                return f"업로드 실패: 파일이 너무 작습니다 (Content-Type: {content_type}, 크기: {len(response.content)} bytes)"
            
            # S3에 업로드할 파일명 생성
            s3_key = f"veo-videos/veo_video_{task_id}.mp4"
            
            # 비디오 데이터를 BytesIO로 래핑
            video_data = io.BytesIO(response.content)
            
            # S3에 업로드
            s3_client.upload_fileobj(
                video_data,
                s3_bucket,
                s3_key,
                ExtraArgs={
                    'ContentType': content_type,
                    'Metadata': {
                        'task_id': task_id,
                        'source': 'google-veo',
                        'original_uri': video_uri
                    }
                }
            )
            
            file_size = len(response.content) / (1024 * 1024)  # MB
            s3_url = f"https://{s3_bucket}.s3.{aws_region}.amazonaws.com/{s3_key}"
            return f"S3 업로드 완료: {s3_url} (크기: {file_size:.2f}MB)"
            
    except NoCredentialsError:
        return "업로드 실패: AWS 자격 증명을 찾을 수 없습니다."
    except ClientError as e:
        return f"S3 업로드 실패: {str(e)}"
    except Exception as e:
        return f"업로드 실패: {str(e)}"


@celery_app.task(bind=True, name="generate_veo_video_async")
def generate_veo_video_task(
    self,
    prompt: str,
    aspect_ratio: str = "16:9",
    model: str = "gemini-1.5-flash",
    timeout_seconds: int = 600
) -> Dict[str, Any]:
    """
    비동기로 Veo 비디오를 생성하는 Celery 태스크
    """
    
    # 진행 상황 업데이트
    self.update_state(
        state="PROGRESS",
        meta={"status": "starting", "message": "비디오 생성을 시작합니다..."}
    )
    
    if not _has_genai:
        return {
            "status": "error",
            "video_uri": None,
            "operation": None,
            "message": None,
            "error": "google-genai 패키지가 필요합니다. pip install google-genai를 실행하세요."
        }

    # Google API Key 확인
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        return {
            "status": "error",
            "video_uri": None,
            "operation": None,
            "message": None,
            "error": "GOOGLE_API_KEY 환경변수가 설정되지 않았습니다."
        }

    try:
        # 진행 상황 업데이트
        self.update_state(
            state="PROGRESS",
            meta={"status": "initializing", "message": "Google GenAI 클라이언트를 초기화합니다..."}
        )
        
        # Google GenAI Client 초기화
        client = genai.Client(api_key=google_api_key)
        
        # 진행 상황 업데이트
        self.update_state(
            state="PROGRESS",
            meta={"status": "generating", "message": f"'{prompt}' 프롬프트로 Veo 3 비디오를 생성중입니다..."}
        )
        
        # Veo 3 비디오 생성 시작
        operation = client.models.generate_videos(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            config=GenerateVideosConfig(
                aspect_ratio=aspect_ratio
                # duration 매개변수는 현재 지원되지 않음
            )
        )
        
        # 비디오 생성 완료까지 대기
        start_time = time.time()
        while not operation.done:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                return {
                    "status": "error",
                    "video_uri": None,
                    "operation": str(operation),
                    "message": None,
                    "error": f"비디오 생성 시간 초과 ({timeout_seconds}초)"
                }
            
            # 진행 상황 업데이트
            progress_message = f"비디오 생성 중... ({int(elapsed_time)}초 경과)"
            self.update_state(
                state="PROGRESS", 
                meta={"status": "generating", "message": progress_message}
            )
            
            time.sleep(15)  # 15초마다 상태 확인
            operation = client.operations.get(operation)
        
        # 비디오 생성 성공 확인
        if not operation.response or not hasattr(operation, 'result') or not operation.result:
            return {
                "status": "error",
                "video_uri": None,
                "operation": str(operation),
                "message": None,
                "error": "비디오 생성에 실패했습니다."
            }
        
        # 생성된 비디오 URI 가져오기
        video_uri = None
        if hasattr(operation.result, 'generated_videos') and operation.result.generated_videos:
            video_obj = operation.result.generated_videos[0]
            if hasattr(video_obj, 'video') and video_obj.video and hasattr(video_obj.video, 'uri'):
                video_uri = video_obj.video.uri
                
        if not video_uri:
            return {
                "status": "error",
                "video_uri": None,
                "operation": str(operation),
                "message": None,
                "error": "생성된 비디오 URI를 가져올 수 없습니다."
            }
        
        # S3에 비디오 업로드
        s3_upload_result = upload_video_to_s3(video_uri, self.request.id)
        
        # 완료 상태 업데이트
        self.update_state(
            state="PROGRESS",
            meta={"status": "completed", "message": "비디오 생성 및 S3 업로드가 완료되었습니다."}
        )
        
        return {
            "status": "completed",
            "video_uri": video_uri,
            "s3_path": s3_upload_result,
            "operation": str(operation),
            "message": f"비디오가 성공적으로 생성되고 S3에 업로드되었습니다: {s3_upload_result}",
            "error": None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "video_uri": None,
            "operation": None,
            "message": None,
            "error": f"Google GenAI 호출 중 오류 발생: {str(e)}"
        } 