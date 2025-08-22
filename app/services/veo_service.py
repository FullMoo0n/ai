"""
Veo 비디오 생성 서비스 모듈

Google Veo 3 API를 사용하여 Gemini가 생성한 프롬프트로 수어 비디오를 생성하는 서비스
"""

import os
import time
import logging
import asyncio
import boto3
import httpx
import io
from typing import Optional, Dict, Any, List
from datetime import datetime
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)

# Google GenAI SDK 사용 (새로운 API)
try:
    from google import genai
    from google.genai.types import GenerateVideosConfig, Image
    _has_genai = True
except ImportError:
    _has_genai = False


class VeoServiceError(Exception):
    """Veo 서비스 에러"""
    pass


class VeoAPIKeyError(VeoServiceError):
    """Veo API 키 에러"""
    pass


class VeoGenerationError(VeoServiceError):
    """Veo 생성 에러"""
    pass


class VeoService:
    """Veo 비디오 생성 서비스"""
    
    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 600):
        """
        Veo 서비스 초기화
        
        Args:
            api_key: Google API 키 (없으면 환경변수에서 가져옴)
            timeout_seconds: 비디오 생성 대기 시간 (기본: 10분)
        """
        if not _has_genai:
            raise VeoServiceError("google-genai 패키지가 필요합니다. pip install google-genai를 실행하세요.")
        
        self.api_key = api_key or os.getenv('VEO_API_KEY')
        if not self.api_key:
            raise VeoAPIKeyError("VEO_API_KEY 환경변수가 설정되지 않았습니다.")
        
        self.timeout_seconds = timeout_seconds
        self._initialize_genai()
        self._initialize_s3()
    
    def _initialize_genai(self):
        """Google GenAI 초기화"""
        try:
            # 새로운 API 클라이언트 초기화
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Google GenAI 클라이언트 초기화 완료")
        except Exception as e:
            raise VeoAPIKeyError(f"Google GenAI 클라이언트 초기화 실패: {str(e)}")
    
    def _initialize_s3(self):
        """S3 클라이언트 초기화"""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=os.getenv("AWS_REGION", "ap-northeast-2")
            )
            self.s3_bucket = os.getenv("S3_BUCKET_NAME")
            logger.info("S3 클라이언트 초기화 완료")
        except Exception as e:
            logger.warning(f"S3 클라이언트 초기화 실패: {str(e)}. S3 업로드 기능이 비활성화됩니다.")
            self.s3_client = None
            self.s3_bucket = None
    
    async def upload_video_to_s3(self, video_uri: str, task_id: str) -> str:
        """
        Google API 비디오 URI를 S3에 업로드합니다.
        """
        if not self.s3_client or not self.s3_bucket:
            logger.warning("S3 클라이언트가 초기화되지 않았습니다.")
            return video_uri  # 원본 URI 반환
        
        try:
            # HTTP 클라이언트로 비디오 다운로드 (API 키를 쿼리 파라미터로)
            if "?" in video_uri:
                download_url = f"{video_uri}&key={self.api_key}"
            else:
                download_url = f"{video_uri}?key={self.api_key}"
            
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
                response = await client.get(download_url)
                response.raise_for_status()
                
                # Content-Type 확인
                content_type = response.headers.get('content-type', 'video/mp4')
                if len(response.content) < 1000:
                    logger.warning(f"파일이 너무 작습니다 (크기: {len(response.content)} bytes)")
                    return video_uri  # 원본 URI 반환
                
                # S3에 업로드할 파일명 생성
                s3_key = f"veo-videos/veo_video_{task_id}_{int(time.time())}.mp4"
                
                # 비디오 데이터를 BytesIO로 래핑
                video_data = io.BytesIO(response.content)
                
                # S3에 업로드
                self.s3_client.upload_fileobj(
                    video_data,
                    self.s3_bucket,
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
                s3_url = f"https://{self.s3_bucket}.s3.{os.getenv('AWS_REGION', 'ap-northeast-2')}.amazonaws.com/{s3_key}"
                logger.info(f"✅ S3 업로드 완료: {s3_url} (크기: {file_size:.2f}MB)")
                return s3_url
                
        except Exception as e:
            logger.error(f"❌ S3 업로드 실패: {str(e)}")
            return video_uri  # 실패 시 원본 URI 반환
    
    async def generate_sign_video(
        self, 
        prompt: str, 
        aspect_ratio: str = "16:9",
        task_id: Optional[str] = None,
        image_gcs_uri: Optional[str] = None,
        output_gcs_uri: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Gemini 프롬프트를 사용하여 Veo3로 수어 비디오 생성
        
        Args:
            prompt: Gemini가 생성한 수어 비디오 프롬프트
            aspect_ratio: 화면 비율 (기본: 16:9)
            task_id: 작업 ID (로깅용)
            image_gcs_uri: 참조 이미지 GCS URI (선택사항)
            output_gcs_uri: 출력 비디오 GCS URI (선택사항)
            
        Returns:
            Dict: 생성 결과
                - status: "success" | "error" | "mock"
                - video_url: 생성된 비디오 URL
                - prompt: 사용된 프롬프트
                - task_id: 작업 ID
                - created_at: 생성 시간
                - note: 추가 정보
        """
        try:
            logger.info(f"🎬 Veo3 수어 비디오 생성 시작: {task_id or 'unknown'}")
            logger.info(f"📝 프롬프트 길이: {len(prompt)}자")
            logger.info(f"📐 화면 비율: {aspect_ratio}")
            
            # Veo3 API 호출 시도
            try:
                video_result = await self._call_veo3_api(
                    prompt, aspect_ratio, task_id, image_gcs_uri, output_gcs_uri
                )
                if video_result and video_result.get('status') == 'success':
                    logger.info(f"✅ Veo3 API 호출 성공: {task_id or 'unknown'}")
                    
                    # Google API URI를 S3에 업로드
                    original_url = video_result.get('video_url')
                    if original_url and original_url.startswith('https://generativelanguage.googleapis.com'):
                        logger.info(f"📤 Google API URI를 S3에 업로드 중: {task_id or 'unknown'}")
                        s3_url = await self.upload_video_to_s3(original_url, task_id or 'unknown')
                        video_result['video_url'] = s3_url
                        video_result['original_google_uri'] = original_url
                        logger.info(f"✅ S3 업로드 완료, URL 업데이트: {s3_url}")
                    
                    return video_result
                else:
                    logger.warning(f"⚠️ Veo3 API 응답이 예상과 다름: {video_result}")
                    return self._create_mock_result(prompt, task_id, "Veo3 API 응답이 예상과 다름")
                    
            except Exception as e:
                logger.error(f"❌ Veo3 API 호출 실패: {str(e)}")
                return self._create_mock_result(prompt, task_id, f"Veo3 API 오류: {str(e)}")
                
        except Exception as e:
            logger.error(f"❌ 수어 비디오 생성 중 예상치 못한 오류: {str(e)}")
            return self._create_mock_result(prompt, task_id, f"예상치 못한 오류: {str(e)}")
    
    async def _call_veo3_api(
        self, 
        prompt: str, 
        aspect_ratio: str, 
        task_id: Optional[str],
        image_gcs_uri: Optional[str] = None,
        output_gcs_uri: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Veo3 API를 실제로 호출합니다 (새로운 API 형식).
        
        Args:
            prompt: 비디오 생성 프롬프트
            aspect_ratio: 화면 비율
            task_id: 작업 ID
            image_gcs_uri: 참조 이미지 GCS URI
            output_gcs_uri: 출력 비디오 GCS URI
            
        Returns:
            Dict: Veo3 API 응답 또는 None
        """
        try:
            logger.info(f"🚀 Veo3 API 호출 시작: {task_id or 'unknown'}")
            
            # 기본 설정값들
            model_name = os.getenv('VEO_MODEL', 'veo-3.0-generate-preview')
            
            # 이미지 설정 (선택사항)
            image_config = None
            if image_gcs_uri:
                image_config = Image(
                    gcs_uri=image_gcs_uri,
                    mime_type="image/png"  # 기본값, 필요시 수정
                )
                logger.info(f"🖼️ 참조 이미지 사용: {image_gcs_uri}")
            
            # 출력 설정 (선택사항)
            if not output_gcs_uri:
                output_gcs_uri = os.getenv('VEO_OUTPUT_GCS_URI')
                if not output_gcs_uri:
                    logger.warning("⚠️ VEO_OUTPUT_GCS_URI가 설정되지 않았습니다. 기본값을 사용합니다.")
            
            # Veo3 비디오 생성 요청
            operation = self.client.models.generate_videos(
                model=model_name,
                prompt=prompt,  # 텍스트 프롬프트
                image=image_config,  # 선택사항
                config=GenerateVideosConfig(
                    aspect_ratio=aspect_ratio,
                ),
            )
            
            logger.info(f"⏳ Veo3 비디오 생성 작업 시작: {operation.name}")
            
            # 비동기로 작업 완료 대기
            video_result = await self._wait_for_operation_completion(operation, task_id)
            
            if video_result:
                logger.info(f"✅ Veo3 비디오 생성 성공: {task_id or 'unknown'}")
                return video_result
            else:
                logger.warning(f"⚠️ Veo3 비디오 생성 실패")
                return None
                
        except Exception as e:
            logger.error(f"❌ Veo3 API 호출 중 오류: {str(e)}")
            raise
    
    async def _wait_for_operation_completion(
        self, 
        operation, 
        task_id: Optional[str],
        check_interval: int = 15
    ) -> Optional[Dict[str, Any]]:
        """
        Veo3 작업 완료를 기다립니다.
        
        Args:
            operation: Veo3 작업 객체
            task_id: 작업 ID
            check_interval: 상태 확인 간격 (초)
            
        Returns:
            Dict: 완료된 비디오 결과 또는 None
        """
        try:
            start_time = time.time()
            logger.info(f"⏰ Veo3 작업 완료 대기 시작: {task_id or 'unknown'}")
            
            while not operation.done:
                # 타임아웃 체크
                if time.time() - start_time > self.timeout_seconds:
                    logger.error(f"⏰ Veo3 작업 타임아웃: {task_id or 'unknown'}")
                    return None
                
                # 상태 확인 간격만큼 대기
                await asyncio.sleep(check_interval)
                
                # 작업 상태 업데이트
                try:
                    operation = self.client.operations.get(operation)
                    logger.info(f"📊 Veo3 작업 상태: {operation.name} - 진행중...")
                except Exception as e:
                    logger.warning(f"⚠️ Veo3 작업 상태 확인 실패: {str(e)}")
                    continue
            
            # 작업 완료 확인
            if operation.response and hasattr(operation.result, 'generated_videos'):
                video_uri = operation.result.generated_videos[0].video.uri
                logger.info(f"🎬 Veo3 비디오 생성 완료: {video_uri}")
                
                return {
                    'status': 'success',
                    'video_url': video_uri,
                    'video_data': {
                        'uri': video_uri,
                        'operation_name': operation.name
                    },
                    'prompt': getattr(operation, 'prompt', ''),
                    'task_id': task_id,
                    'created_at': datetime.now().isoformat(),
                    'note': 'Veo3 API 성공 - 새로운 API 형식'
                }
            else:
                logger.warning(f"⚠️ Veo3 작업 응답에 비디오 데이터가 없음: {operation}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Veo3 작업 완료 대기 중 오류: {str(e)}")
            return None
    
    def _create_mock_result(self, prompt: str, task_id: Optional[str], note: str) -> Dict[str, Any]:
        """
        Veo3 API 호출 실패 시 모의 결과를 생성합니다.
        
        Args:
            prompt: 사용된 프롬프트
            task_id: 작업 ID
            note: 실패 이유
            
        Returns:
            Dict: 모의 결과
        """
        timestamp = datetime.now().strftime('%H%M%S')
        mock_filename = f"mock_video_{task_id or 'unknown'}_{timestamp}.mp4"
        mock_url = f"https://ddo123.s3.ap-northeast-2.amazonaws.com/videos/{mock_filename}"
        
        logger.info(f"🔄 모의 비디오 결과 생성: {mock_url}")
        
        return {
            'status': 'mock',
            'video_url': mock_url,
            'prompt': prompt,
            'task_id': task_id,
            'created_at': datetime.now().isoformat(),
            'note': note
        }
    
    async def generate_multiple_videos(
        self, 
        prompts: Dict[str, str], 
        aspect_ratio: str = "16:9",
        image_gcs_uri: Optional[str] = None,
        output_gcs_uri: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        여러 프롬프트로 여러 비디오를 생성합니다.
        
        Args:
            prompts: 문장별 프롬프트 딕셔너리
            aspect_ratio: 화면 비율
            image_gcs_uri: 참조 이미지 GCS URI (선택사항)
            output_gcs_uri: 출력 비디오 GCS URI (선택사항)
            
        Returns:
            Dict: 문장별 생성 결과
        """
        logger.info(f"🎬 다중 비디오 생성 시작: {len(prompts)}개")
        
        results = {}
        for sentence, prompt in prompts.items():
            task_id = f"veo_{hash(sentence) % 10000:04d}"
            result = await self.generate_sign_video(
                prompt, aspect_ratio, task_id, image_gcs_uri, output_gcs_uri
            )
            results[sentence] = result
            
            # API 호출 간격 조절
            await asyncio.sleep(1)
        
        logger.info(f"✅ 다중 비디오 생성 완료: {len(results)}개")
        return results
    
    def validate_prompt(self, prompt: str) -> bool:
        """
        프롬프트 유효성을 검증합니다.
        
        Args:
            prompt: 검증할 프롬프트
            
        Returns:
            bool: 유효성 여부
        """
        if not prompt or not prompt.strip():
            return False
        
        # 최소 길이 검증
        if len(prompt.strip()) < 50:
            return False
        
        # 필수 키워드 검증
        required_keywords = ['sign language', 'Korean', 'video']
        prompt_lower = prompt.lower()
        
        for keyword in required_keywords:
            if keyword not in prompt_lower:
                return False
        
        return True
    
    def get_prompt_statistics(self, prompt: str) -> Dict[str, Any]:
        """
        프롬프트 통계 정보를 제공합니다.
        
        Args:
            prompt: 분석할 프롬프트
            
        Returns:
            Dict: 통계 정보
        """
        return {
            'length': len(prompt),
            'word_count': len(prompt.split()),
            'line_count': len(prompt.split('\n')),
            'has_sign_details': 'Sign Language Details:' in prompt,
            'has_video_requirements': 'Video Requirements:' in prompt,
            'has_instructions': 'Key Instructions:' in prompt
        }


# 편의 함수들
async def generate_sign_video(
    prompt: str, 
    aspect_ratio: str = "16:9",
    image_gcs_uri: Optional[str] = None,
    output_gcs_uri: Optional[str] = None
) -> Dict[str, Any]:
    """
    편의 함수: 수어 비디오 생성
    
    Args:
        prompt: Gemini가 생성한 프롬프트
        aspect_ratio: 화면 비율
        image_gcs_uri: 참조 이미지 GCS URI (선택사항)
        output_gcs_uri: 출력 비디오 GCS URI (선택사항)
        
    Returns:
        Dict: 생성 결과
    """
    service = VeoService()
    return await service.generate_sign_video(prompt, aspect_ratio, image_gcs_uri=image_gcs_uri, output_gcs_uri=output_gcs_uri)


async def generate_multiple_sign_videos(
    prompts: Dict[str, str], 
    aspect_ratio: str = "16:9",
    image_gcs_uri: Optional[str] = None,
    output_gcs_uri: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    편의 함수: 여러 수어 비디오 생성
    
    Args:
        prompts: 문장별 프롬프트
        aspect_ratio: 화면 비율
        image_gcs_uri: 참조 이미지 GCS URI (선택사항)
        output_gcs_uri: 출력 비디오 GCS URI (선택사항)
        
    Returns:
        Dict: 문장별 생성 결과
    """
    service = VeoService()
    return await service.generate_multiple_videos(
        prompts, aspect_ratio, image_gcs_uri=image_gcs_uri, output_gcs_uri=output_gcs_uri
    )