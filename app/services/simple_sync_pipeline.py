"""
간소화된 동기 파이프라인 모듈

문장 분할 없이 전체 텍스트를 한번에 처리하는 파이프라인
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# 기존 서비스들 임포트
from .tokenizer import tokenize
from .prompt_template import get_default_prompt_manager
from .veo_service import generate_sign_video

logger = logging.getLogger(__name__)


class SimpleSyncPipelineError(Exception):
    """간소화된 동기 파이프라인 에러"""
    pass


@dataclass
class PipelineStep:
    """파이프라인 단계 정보"""
    name: str
    status: str = "pending"  # pending, processing, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    data: Optional[Any] = None


class SimpleSyncPipeline:
    """간소화된 동기 처리 파이프라인 클래스"""
    
    def __init__(self, task_id: Optional[str] = None):
        """
        파이프라인 초기화
        
        Args:
            task_id: 작업 ID
        """
        self.task_id = task_id or f"simple_sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.pipeline_status = "initialized"
        self.current_step = None
        self.final_result = None
        
        # 단계별 상태 관리
        self.steps = {
            "ocr": PipelineStep("ocr"),
            "tokenization": PipelineStep("tokenization"),
            "prompt_generation": PipelineStep("prompt_generation"),
            "video_generation": PipelineStep("video_generation"),
            "finalization": PipelineStep("finalization")
        }
        
        logger.info(f"간소화된 동기 파이프라인 초기화: {self.task_id}")
    
    def run(self, s3_image_url: str) -> Dict[str, Any]:
        """
        파이프라인 실행
        
        Args:
            s3_image_url: S3 이미지 URL
            
        Returns:
            Dict: 최종 결과
        """
        try:
            logger.info(f"간소화된 동기 파이프라인 시작: {self.task_id}")
            self.pipeline_status = "running"
            
            # 1. OCR 텍스트 추출
            extracted_text = self._step_ocr(s3_image_url)
            
            # 2. 토큰화 및 수어 데이터 조회
            token_result = self._step_tokenize_and_process(extracted_text)
            
            # 3. 프롬프트 생성
            prompt_result = self._step_generate_prompt(token_result)
            
            # 4. 비디오 생성
            video_result = self._step_generate_video(prompt_result)
            
            # 5. 최종 결과 취합
            final_result = self._step_finalize(video_result)
            
            self.pipeline_status = "completed"
            self.final_result = final_result
            
            logger.info(f"간소화된 동기 파이프라인 완료: {self.task_id}")
            return final_result
            
        except Exception as e:
            self.pipeline_status = "failed"
            logger.error(f"간소화된 동기 파이프라인 실패: {self.task_id}, 오류: {e}")
            raise SimpleSyncPipelineError(f"파이프라인 실행 실패: {str(e)}")
    
    def _step_ocr(self, s3_image_url: str) -> str:
        """1단계: OCR 텍스트 추출"""
        step = self.steps["ocr"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "ocr"
        
        try:
            logger.info(f"OCR 단계 시작: {self.task_id}")
            
            # 간단하게 모의 OCR 결과 생성
            extracted_text = "안녕하세요. 수어 비디오를 생성합니다. 전체 텍스트를 한번에 처리합니다."
            
            step.data = {'text': extracted_text}
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"✅ OCR 완료: {self.task_id}")
            logger.info(f"📝 추출된 텍스트: '{extracted_text}'")
            return extracted_text
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise
    
    def _step_tokenize_and_process(self, text: str) -> Dict[str, Any]:
        """2단계: 전체 텍스트 토큰화 및 수어 데이터 조회"""
        step = self.steps["tokenization"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "tokenization"
        
        try:
            logger.info(f"토큰화 및 수어 데이터 조회 단계 시작: {self.task_id}")
            logger.info(f"🔤 원본 텍스트: '{text}'")
            
            # 전체 텍스트 토큰화
            tokens = tokenize(text)
            logger.info(f"🔗 토큰화 결과: {tokens} (총 {len(tokens)}개)")
            
            # 각 토큰에 대해 수어 데이터 조회
            sign_data = []
            for token in tokens:
                try:
                    # 간단하게 모의 수어 설명 생성
                    description = f"{token}에 대한 수어 표현"
                    
                    sign_data.append({
                        'word': token,
                        'description': description,
                        'culture_data': {'description': description}
                    })
                except Exception as e:
                    logger.warning(f"토큰 '{token}' 수어 데이터 조회 실패: {e}")
                    sign_data.append({
                        'word': token,
                        'description': f'{token}에 대한 수어 표현',
                        'culture_data': None,
                        'error': str(e)
                    })
            
            logger.info(f"📊 수어 데이터 조회 완료: {len(sign_data)}개")
            
            # 결과 구조
            result = {
                'text': text,
                'tokens': tokens,
                'sign_data': sign_data
            }
            
            step.data = result
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"✅ 토큰화 및 수어 데이터 조회 완료: {self.task_id}")
            return result
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise
    
    def _step_generate_prompt(self, token_result: Dict[str, Any]) -> Dict[str, Any]:
        """3단계: 전체 텍스트 기반 프롬프트 생성"""
        step = self.steps["prompt_generation"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "prompt_generation"
        
        try:
            logger.info(f"프롬프트 생성 단계 시작: {self.task_id}")
            
            text = token_result['text']
            tokens = token_result['tokens']
            sign_data = token_result['sign_data']
            
            # 전체 텍스트에 대한 프롬프트 생성 (Gemini AI 사용)
            prompt_manager = get_default_prompt_manager()
            video_prompt = prompt_manager.generate_video_prompt(text, sign_data)
            
            logger.info(f"📝 생성된 프롬프트 길이: {len(video_prompt)}자")
            logger.info(f"🔍 프롬프트 내용 (처음 200자): '{video_prompt[:200]}...'")
            
            # 결과 구조
            result = {
                'text': text,
                'tokens': tokens,
                'sign_data': sign_data,
                'video_prompt': video_prompt
            }
            
            step.data = result
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"✅ 프롬프트 생성 완료: {self.task_id}")
            return result
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            logger.error(f"❌ 프롬프트 생성 실패: {e}")
            raise
    
    def _step_generate_video(self, prompt_result: Dict[str, Any]) -> Dict[str, Any]:
        """4단계: 비디오 생성"""
        step = self.steps["video_generation"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "video_generation"
        
        try:
            logger.info(f"비디오 생성 단계 시작: {self.task_id}")
            
            text = prompt_result['text']
            video_prompt = prompt_result['video_prompt']
            
            logger.info(f"🎬 비디오 생성 시작")
            logger.info(f"📝 텍스트: '{text}'")
            logger.info(f"🎭 프롬프트 길이: {len(video_prompt)}자")
            
            # VEO 서비스 직접 사용 (동기 처리)
            from .veo_service import VeoService
            veo_service = VeoService()
            
            # 모의 결과 생성 (실제 API 호출은 비동기이므로)
            veo_result = {
                'status': 'mock',
                'video_url': f'https://mock-veo-video.com/video_{self.task_id}.mp4',
                'prompt': video_prompt,
                'task_id': self.task_id,
                'created_at': datetime.now().isoformat(),
                'note': '동기 파이프라인용 모의 결과'
            }
            
            if veo_result['status'] == 'success':
                logger.info(f"✅ 비디오 생성 성공: {veo_result.get('video_url', 'N/A')}")
                video_result = {
                    'text': text,
                    'video_url': veo_result.get('video_url'),
                    'original_google_uri': veo_result.get('original_google_uri'),
                    'video_prompt': video_prompt,
                    'veo_result': veo_result,
                    'status': 'success'
                }
            else:
                logger.warning(f"⚠️ 비디오 생성 실패 또는 모의 결과: {veo_result.get('status')}")
                video_result = {
                    'text': text,
                    'video_url': veo_result.get('video_url'),
                    'video_prompt': video_prompt,
                    'veo_result': veo_result,
                    'status': veo_result.get('status', 'failed')
                }
            
            step.data = video_result
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"✅ 비디오 생성 단계 완료: {self.task_id}")
            return video_result
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            logger.error(f"❌ 비디오 생성 실패: {e}")
            raise
    
    def _step_finalize(self, video_result: Dict[str, Any]) -> Dict[str, Any]:
        """5단계: 최종 결과 취합"""
        step = self.steps["finalization"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "finalization"
        
        try:
            logger.info(f"최종 결과 취합 단계 시작: {self.task_id}")
            
            # 최종 결과 구조
            final_result = {
                'task_id': self.task_id,
                'pipeline_type': 'simple_sync',
                'status': 'completed',
                'text': video_result['text'],
                'video_url': video_result.get('video_url'),
                'original_google_uri': video_result.get('original_google_uri'),
                'video_prompt': video_result['video_prompt'],
                'veo_status': video_result['status'],
                'created_at': datetime.now().isoformat(),
                'steps_summary': {
                    step_name: {
                        'status': step_info.status,
                        'started_at': step_info.started_at.isoformat() if step_info.started_at else None,
                        'completed_at': step_info.completed_at.isoformat() if step_info.completed_at else None,
                        'error': step_info.error
                    }
                    for step_name, step_info in self.steps.items()
                }
            }
            
            step.data = final_result
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"✅ 최종 결과 취합 완료: {self.task_id}")
            logger.info(f"📊 결과 요약:")
            logger.info(f"   📝 텍스트: '{final_result['text']}'")
            logger.info(f"   🎬 비디오 URL: {final_result.get('video_url', 'N/A')}")
            logger.info(f"   📈 VEO 상태: {final_result['veo_status']}")
            
            return final_result
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            logger.error(f"❌ 최종 결과 취합 실패: {e}")
            raise


# 편의 함수
def run_simple_sync_pipeline(s3_image_url: str, task_id: Optional[str] = None) -> Dict[str, Any]:
    """
    간소화된 동기 파이프라인 실행
    
    Args:
        s3_image_url: S3 이미지 URL
        task_id: 작업 ID (선택사항)
        
    Returns:
        Dict: 파이프라인 실행 결과
    """
    pipeline = SimpleSyncPipeline(task_id)
    return pipeline.run(s3_image_url) 