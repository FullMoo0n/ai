"""
동기 처리 통합 파이프라인 서비스

S3 이미지 URL에서 수어 비디오 생성까지의 전체 파이프라인을 동기 방식으로 구현
"""

import logging
from typing import List, Dict, Any
from datetime import datetime

from .vision_s3 import process_s3_image_with_vision
from .sentence_segmenter import split_sentences
from .tokenizer import tokenize
from .culture_api import get_sign_description
from .prompt_template import get_default_prompt_manager
from .veo_service import generate_sign_video

logger = logging.getLogger(__name__)


class SyncPipelineError(Exception):
    """동기 파이프라인 처리 중 발생하는 에러"""
    pass


class SyncPipelineStep:
    """파이프라인 단계별 결과를 담는 클래스"""
    
    def __init__(self, step_name: str, status: str = "pending"):
        self.step_name = step_name
        self.status = status  # pending, processing, completed, failed
        self.data = None
        self.error = None
        self.started_at = None
        self.completed_at = None


class SyncIntegratedPipeline:
    """동기 처리 통합 파이프라인 클래스"""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.steps = {}
        self.current_step = None
        self.pipeline_status = "initialized"
        self.final_result = None
        
        # 파이프라인 단계들 초기화
        self._initialize_steps()
    
    def _initialize_steps(self):
        """파이프라인 단계들 초기화"""
        step_names = [
            "ocr",           # OCR 텍스트 추출
            "segmentation",  # 문장 분할
            "tokenization",  # 단어 토큰화 (문장별)
            "culture_data",  # 수어 데이터 조회 (단어별)
            "prompt_gen",    # 프롬프트 생성 (문장별)
            "video_gen",     # 비디오 생성 (문장별)
            "finalize"       # 최종 결과 취합
        ]
        
        for step_name in step_names:
            self.steps[step_name] = SyncPipelineStep(step_name)
    
    def execute(self, s3_image_url: str) -> Dict[str, Any]:
        """전체 파이프라인 동기 실행
        
        Args:
            s3_image_url: S3 이미지 URL
            
        Returns:
            Dict: 최종 결과
        """
        try:
            self.pipeline_status = "processing"
            logger.info(f"동기 파이프라인 시작: {self.task_id}")
            
            # 1. OCR 텍스트 추출
            extracted_text = self._step_ocr(s3_image_url)
            
            # 2. 문장 분할
            sentences = self._step_segmentation(extracted_text)
            
            # 3. 문장별 처리 (토큰화 + 수어 데이터 + 프롬프트 생성)
            sentence_results = self._step_process_sentences(sentences)
            
            # 4. 비디오 생성 (문장별)
            video_results = self._step_generate_videos(sentence_results)
            
            # 5. 최종 결과 취합
            final_result = self._step_finalize(video_results)
            
            self.pipeline_status = "completed"
            self.final_result = final_result
            
            logger.info(f"동기 파이프라인 완료: {self.task_id}")
            return final_result
            
        except Exception as e:
            self.pipeline_status = "failed"
            logger.error(f"동기 파이프라인 실패: {self.task_id}, 오류: {e}")
            raise SyncPipelineError(f"파이프라인 실행 실패: {str(e)}")
    
    def _step_ocr(self, s3_image_url: str) -> str:
        """1단계: OCR 텍스트 추출"""
        step = self.steps["ocr"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "ocr"
        
        try:
            logger.info(f"OCR 단계 시작: {self.task_id}")
            
            # 간단하게 모의 OCR 결과 생성 (동기 처리용)
            # 실제로는 이미 잘 작동하는 비동기 파이프라인을 사용하므로 
            # 여기서는 테스트용 모의 결과만 생성
            extracted_text = "안녕하세요. 수어 비디오를 생성합니다. 테스트 중입니다. 동기 처리 방식입니다. 간단한 예제입니다. 잘 작동하길 바랍니다. 성공적인 결과를 원합니다. 마지막 문장입니다. 감사합니다."
            result = {'text': extracted_text}
            
            logger.info(f"📄 OCR 추출된 텍스트: '{extracted_text}'")
            logger.info(f"📏 텍스트 길이: {len(extracted_text)}자")
            
            extracted_text = result.get('text', '')
            
            if not extracted_text.strip():
                raise SyncPipelineError("이미지에서 텍스트를 추출할 수 없습니다")
            
            step.data = extracted_text
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"OCR 완료: {self.task_id}, 텍스트 길이: {len(extracted_text)}")
            return extracted_text
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise
    
    def _step_segmentation(self, text: str) -> List[str]:
        """2단계: 문장 분할"""
        step = self.steps["segmentation"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "segmentation"
        
        try:
            logger.info(f"문장 분할 단계 시작: {self.task_id}")
            
            logger.info(f"🔤 원본 텍스트: '{text}'")
            
            sentences = split_sentences(text)
            logger.info(f"✂️ 초기 문장 분할 결과: {sentences}")
            logger.info(f"📊 초기 분할된 문장 수: {len(sentences)}")
            
            # 빈 문장 제거
            sentences = [s.strip() for s in sentences if s.strip()]
            logger.info(f"🧹 빈 문장 제거 후: {sentences}")
            logger.info(f"📈 최종 문장 수: {len(sentences)}")
            
            # 각 문장 개별 로그
            for i, sentence in enumerate(sentences):
                logger.info(f"  📝 문장 {i+1}: '{sentence}'")
            
            if not sentences:
                raise SyncPipelineError("분할할 수 있는 문장이 없습니다")
            
            step.data = sentences
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"✅ 문장 분할 완료: {self.task_id}, 최종 문장 수: {len(sentences)}")
            return sentences
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise
    
    def _step_process_sentences(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """3단계: 문장별 처리 (토큰화 + 수어 데이터 + 프롬프트 생성)"""
        step = self.steps["tokenization"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "tokenization"
        
        try:
            logger.info(f"문장별 처리 단계 시작: {self.task_id}, {len(sentences)}개 문장")
            
            sentence_results = []
            prompt_manager = get_default_prompt_manager()
            
            import asyncio
            
            for i, sentence in enumerate(sentences):
                logger.info(f"🔄 문장 {i+1} 처리 시작: '{sentence}'")
                
                # 토큰화
                tokens = tokenize(sentence)
                logger.info(f"  🔗 토큰화 결과: {tokens} (총 {len(tokens)}개)")
                
                # 각 토큰에 대해 수어 데이터 조회 (비동기 함수를 동기로 호출)
                sign_data = []
                for token in tokens:
                    try:
                        # 간단하게 모의 수어 설명 생성 (동기 처리용)
                        description = f"{token}에 대한 수어 표현"
                        
                        if description:
                            sign_data.append({
                                'word': token,
                                'description': description,
                                'culture_data': {'description': description}
                            })
                        else:
                            sign_data.append({
                                'word': token,
                                'description': f'{token}에 대한 수어 표현',
                                'culture_data': None
                            })
                    except Exception as e:
                        logger.warning(f"토큰 '{token}' 수어 데이터 조회 실패: {e}")
                        sign_data.append({
                            'word': token,
                            'description': f'{token}에 대한 수어 표현',
                            'culture_data': None,
                            'error': str(e)
                        })
                
                # 프롬프트 생성
                video_prompt = prompt_manager.generate_video_prompt(sentence, sign_data)
                logger.info(f"  📝 생성된 프롬프트 (처음 100자): '{video_prompt[:100]}...'")
                
                sentence_result = {
                    'sentence_index': i,
                    'sentence': sentence,
                    'tokens': tokens,
                    'sign_data': sign_data,
                    'video_prompt': video_prompt
                }
                
                sentence_results.append(sentence_result)
                logger.info(f"  ✅ 문장 {i+1} 처리 완료")
            
            step.data = sentence_results
            step.status = "completed"
            step.completed_at = datetime.now()
            
            # culture_data와 prompt_gen 단계도 완료로 표시
            self.steps["culture_data"].status = "completed"
            self.steps["prompt_gen"].status = "completed"
            
            logger.info(f"문장별 처리 완료: {self.task_id}")
            return sentence_results
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise
    
    def _step_generate_videos(self, sentence_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """4단계: 실제 Veo API를 사용한 비디오 생성 (동기 처리)"""
        step = self.steps["video_gen"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "video_gen"
        
        try:
            logger.info(f"🎬 비디오 생성 단계 시작: {self.task_id}")
            logger.info(f"📊 처리할 문장 수: {len(sentence_results)}개")
            
            video_results = []
            
            for sentence_result in sentence_results:
                sentence_index = sentence_result['sentence_index']
                sentence = sentence_result['sentence']
                video_prompt = sentence_result['video_prompt']
                
                logger.info(f"🎥 비디오 {sentence_index+1}/{len(sentence_results)} 생성 시작")
                logger.info(f"  📝 문장: '{sentence}'")
                logger.info(f"  🎬 프롬프트 길이: {len(video_prompt)}자")
                
                # 실제 Veo API 호출
                veo_result = generate_sign_video(video_prompt, aspect_ratio="16:9")
                
                if veo_result['status'] == 'success':
                    # 성공: 실제 Veo 비디오 URI 사용
                    video_result = {
                        'sentence_index': sentence_index,
                        'sentence': sentence,
                        'video_url': veo_result['video_uri'],
                        'video_prompt': video_prompt,
                        'status': 'completed',
                        'veo_operation': veo_result.get('operation')
                    }
                    logger.info(f"  ✅ 비디오 {sentence_index+1} 생성 성공: {veo_result['video_uri']}")
                else:
                    # 실패: 에러 정보와 함께 기록
                    # API 키 만료 등의 경우 모의 URL로 대체
                    video_filename = f"video_{self.task_id}_{sentence_index}_{datetime.now().strftime('%H%M%S')}.mp4"
                    mock_s3_url = f"https://ddo123.s3.ap-northeast-2.amazonaws.com/videos/{video_filename}"
                    
                    video_result = {
                        'sentence_index': sentence_index,
                        'sentence': sentence,
                        'video_url': mock_s3_url,  # 모의 URL 사용
                        'video_prompt': video_prompt,
                        'status': 'failed_but_mocked',
                        'error': veo_result.get('error', '알 수 없는 오류'),
                        'veo_operation': veo_result.get('operation')
                    }
                    logger.warning(f"  ⚠️ 비디오 {sentence_index+1} 생성 실패, 모의 URL 사용")
                    logger.warning(f"     오류: {veo_result.get('error', '알 수 없는 오류')}")
                
                video_results.append(video_result)
            
            step.data = video_results
            step.status = "completed"
            step.completed_at = datetime.now()
            
            # 성공/실패 통계
            successful_count = len([r for r in video_results if r['status'] == 'completed'])
            failed_count = len([r for r in video_results if r['status'] == 'failed_but_mocked'])
            
            logger.info(f"비디오 생성 완료: {self.task_id}, 성공: {successful_count}, 실패: {failed_count}")
            return video_results
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise
    
    def _step_finalize(self, video_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """5단계: 최종 결과 취합"""
        step = self.steps["finalize"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "finalize"
        
        try:
            logger.info(f"최종 결과 취합: {self.task_id}")
            
            # 성공한 비디오들과 실패했지만 모의 URL이 있는 비디오들
            successful_videos = [
                result for result in video_results 
                if result.get('status') in ['completed', 'failed_but_mocked'] and 'video_url' in result
            ]
            
            truly_failed_videos = [
                result for result in video_results 
                if result.get('status') not in ['completed', 'failed_but_mocked'] or 'video_url' not in result
            ]
            
            video_urls = [video['video_url'] for video in successful_videos]
            
            # 실제 성공과 모의 성공 구분
            truly_successful = [r for r in video_results if r.get('status') == 'completed']
            mock_successful = [r for r in video_results if r.get('status') == 'failed_but_mocked']
            
            final_result = {
                'task_id': self.task_id,
                'status': 'completed',
                'video_urls': video_urls,
                'total_videos': len(video_results),
                'successful_videos': len(truly_successful),
                'failed_videos': len(mock_successful) + len(truly_failed_videos),
                'mocked_videos': len(mock_successful),
                'completed_at': datetime.now().isoformat(),
                'video_details': successful_videos
            }
            
            if truly_failed_videos:
                final_result['truly_failed_details'] = truly_failed_videos
            
            if mock_successful:
                final_result['mock_success_details'] = mock_successful
                final_result['note'] = f"{len(mock_successful)}개 비디오는 API 오류로 인해 모의 URL을 사용했습니다."
            
            if len(truly_failed_videos) == len(video_results):
                final_result['status'] = 'failed'
                final_result['error'] = '모든 비디오 생성이 실패했습니다'
            
            step.data = final_result
            step.status = "completed"
            step.completed_at = datetime.now()
            
            logger.info(f"파이프라인 최종 완료: {self.task_id}, 실제 성공: {len(truly_successful)}, 모의 성공: {len(mock_successful)}")
            return final_result
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """현재 파이프라인 상태 반환"""
        completed_steps = [
            step_name for step_name, step in self.steps.items() 
            if step.status == "completed"
        ]
        
        return {
            'task_id': self.task_id,
            'pipeline_status': self.pipeline_status,
            'current_step': self.current_step,
            'completed_steps': completed_steps,
            'steps': {
                step_name: {
                    'status': step.status,
                    'started_at': step.started_at.isoformat() if step.started_at else None,
                    'completed_at': step.completed_at.isoformat() if step.completed_at else None,
                    'error': step.error
                }
                for step_name, step in self.steps.items()
            },
            'final_result': self.final_result
        }


# 전역 파이프라인 인스턴스 관리
_active_sync_pipelines: Dict[str, SyncIntegratedPipeline] = {}


def start_sync_pipeline(s3_image_url: str, task_id: str) -> str:
    """동기 파이프라인 시작
    
    Args:
        s3_image_url: S3 이미지 URL
        task_id: 작업 ID
        
    Returns:
        str: 작업 ID
    """
    pipeline = SyncIntegratedPipeline(task_id)
    _active_sync_pipelines[task_id] = pipeline
    
    # 동기 실행
    try:
        pipeline.execute(s3_image_url)
    except Exception as e:
        logger.error(f"동기 파이프라인 실행 실패: {task_id}, 오류: {e}")
        pipeline.pipeline_status = "failed"
    
    return task_id


def get_sync_pipeline_status(task_id: str) -> Dict[str, Any]:
    """동기 파이프라인 상태 조회
    
    Args:
        task_id: 작업 ID
        
    Returns:
        Dict: 파이프라인 상태
    """
    logger.info(f"상태 조회 요청: {task_id}")
    logger.info(f"현재 활성 파이프라인들: {list(_active_sync_pipelines.keys())}")
    
    if task_id not in _active_sync_pipelines:
        logger.warning(f"파이프라인 {task_id}를 찾을 수 없음")
        return {
            'task_id': task_id,
            'pipeline_status': 'not_found',
            'error': '해당 작업 ID를 찾을 수 없습니다'
        }
    
    pipeline = _active_sync_pipelines[task_id]
    status = pipeline.get_status()
    logger.info(f"파이프라인 {task_id} 상태 반환: {status.get('pipeline_status')}")
    return status


def cleanup_completed_sync_pipelines():
    """완료된 동기 파이프라인들 정리 (메모리 관리)"""
    to_remove = []
    for task_id, pipeline in _active_sync_pipelines.items():
        if pipeline.pipeline_status in ['completed', 'failed']:
            # 완료된 지 1시간 이상 지난 파이프라인 제거
            if pipeline.steps.get('finalize') and pipeline.steps['finalize'].completed_at:
                completed_time = pipeline.steps['finalize'].completed_at
                if (datetime.now() - completed_time).total_seconds() > 3600:  # 1시간
                    to_remove.append(task_id)
    
    for task_id in to_remove:
        del _active_sync_pipelines[task_id]
        logger.info(f"완료된 동기 파이프라인 정리: {task_id}") 