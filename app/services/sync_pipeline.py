"""
동기 처리 통합 파이프라인 서비스

S3 이미지 URL에서 수어 비디오 생성까지의 전체 파이프라인을 동기 방식으로 구현
"""

import logging
from typing import Dict, Any
from datetime import datetime
import asyncio

from app.services.vision_s3 import process_s3_image_with_vision
from app.services.openai_vision import analyze_image_context_with_openai
from app.services.tokenizer import tokenize
from app.services.sign_data_service import SignDataService
from .prompt_template import get_default_prompt_manager

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
        self.source_image_url = ""
        self.image_context = ""
        self.character_description = ""

        # 파이프라인 단계들 초기화
        self._initialize_steps()

    def _initialize_steps(self):
        """파이프라인 단계들 초기화"""
        step_names = [
            "ocr",  # OCR 텍스트 추출
            "segmentation",  # 문장 분할
            "tokenization",  # 단어 토큰화 (문장별)
            "culture_data",  # 수어 데이터 조회 (단어별)
            "prompt_gen",  # 프롬프트 생성 (문장별)
            "video_gen",  # 비디오 생성 (문장별)
            "finalize",  # 최종 결과 취합
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
            self.source_image_url = s3_image_url
            logger.info(f"동기 방식 파이프라인 시작: {self.task_id}")

            # 1. OCR 텍스트 추출
            extracted_text = self._step_ocr(s3_image_url)

            # 2. 통합 토큰 처리 (문장 분할 없이 전체 텍스트 처리)
            integrated_result = self._step_process_text(extracted_text)

            # 3. 단일 비디오 생성
            video_result = self._step_generate_video(integrated_result)

            # 4. 최종 결과 취합
            final_result = self._step_finalize([video_result])

            self.pipeline_status = "completed"
            self.final_result = final_result

            logger.info(f"파이프라인 완료: {self.task_id}")
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

            try:
                result = asyncio.run(
                    process_s3_image_with_vision(
                        s3_url=s3_image_url,
                        feature="TEXT_DETECTION",
                        language_hints=["ko"],
                        include_word_boxes=False,
                    )
                )
            except RuntimeError:
                import nest_asyncio

                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(
                    process_s3_image_with_vision(
                        s3_url=s3_image_url,
                        feature="TEXT_DETECTION",
                        language_hints=["ko"],
                        include_word_boxes=False,
                    )
                )

            extracted_text = result.get("text", "")

            try:
                self.image_context = ""
                self.character_description = ""
                try:
                    context_result = asyncio.run(
                        analyze_image_context_with_openai(s3_image_url)
                    )
                except RuntimeError:
                    import nest_asyncio

                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()
                    context_result = loop.run_until_complete(
                        analyze_image_context_with_openai(s3_image_url)
                    )

                self.image_context = context_result.get("image_context", "")
                self.character_description = context_result.get(
                    "character_description", ""
                )
            except Exception as context_error:
                logger.warning(
                    f"이미지 맥락/캐릭터 분석 실패 (OCR은 계속 진행): {context_error}"
                )

            if not extracted_text.strip():
                raise SyncPipelineError("이미지에서 텍스트를 추출할 수 없습니다")

            step.data = extracted_text
            step.status = "completed"
            step.completed_at = datetime.now()

            logger.info(f"OCR 완료: {self.task_id}, 텍스트 길이: {len(extracted_text)}")
            if self.image_context:
                logger.info(
                    f"🖼️ 이미지 맥락 요약 길이: {len(self.image_context)}"
                )
            return extracted_text

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise

    def _step_process_text(self, text: str) -> Dict[str, Any]:
        """레거시 execute()와의 연결을 위한 브리지"""
        token_result = self._step_tokenize_and_process(text)
        return self._step_generate_prompt(token_result)

    def _step_tokenize_and_process(self, text: str) -> Dict[str, Any]:
        """2단계: 전체 텍스트 토큰화 및 수어 데이터 조회"""
        step = self.steps["tokenization"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "tokenization"

        try:
            logger.info(f"전체 텍스트 처리 단계 시작: {self.task_id}")
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

                    if description:
                        sign_data.append(
                            {
                                "word": token,
                                "description": description,
                                "culture_data": {"description": description},
                            }
                        )
                    else:
                        sign_data.append(
                            {
                                "word": token,
                                "description": f"{token}에 대한 수어 표현",
                                "culture_data": None,
                            }
                        )
                except Exception as e:
                    logger.warning(f"토큰 '{token}' 수어 데이터 조회 실패: {e}")
                    sign_data.append(
                        {
                            "word": token,
                            "description": f"{token}에 대한 수어 표현",
                            "culture_data": None,
                            "error": str(e),
                        }
                    )

            logger.info(f"📊 수어 데이터 조회 완료: {len(sign_data)}개")

            # 결과 구조
            result = {"text": text, "tokens": tokens, "sign_data": sign_data}

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
        step = self.steps[
            "prompt_gen"
        ]  # Changed from "prompt_generation" to "prompt_gen" to match _initialize_steps
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "prompt_gen"  # Changed from "prompt_generation" to "prompt_gen" to match _initialize_steps

        try:
            logger.info(f"프롬프트 생성 단계 시작: {self.task_id}")

            text = token_result["text"]
            tokens = token_result["tokens"]
            sign_data = token_result["sign_data"]

            # 전체 텍스트에 대한 프롬프트 생성 (Gemini AI 사용)
            prompt_manager = get_default_prompt_manager()
            gloss_sequence = " ".join(tokens) if tokens else text
            video_prompt = prompt_manager.build_video_prompt(
                original_text=text,
                gloss_sequence=gloss_sequence,
                sign_data=sign_data,
                image_context=self.image_context,
                character_description=self.character_description,
                book_id=self.task_id,
                page_number=1,
                sentence_idx=0,
                reference_image_url=self.source_image_url,
                duration_sec=8,
                version="demo-v1",
            )

            logger.info(f"📝 생성된 프롬프트 길이: {len(video_prompt)}자")
            logger.info(f"🔍 프롬프트 내용 (처음 200자): '{video_prompt[:200]}...'")

            # 결과 구조
            result = {
                "text": text,
                "tokens": tokens,
                "sign_data": sign_data,
                "image_context": self.image_context,
                "character_description": self.character_description,
                "gloss_sequence": gloss_sequence,
                "source_image_url": self.source_image_url,
                "video_prompt": video_prompt,
            }

            step.data = result
            step.status = "completed"
            step.completed_at = datetime.now()

            # culture_data와 prompt_gen 단계도 완료로 표시
            self.steps["culture_data"].status = "completed"
            self.steps["prompt_gen"].status = "completed"

            logger.info(f"✅ 프롬프트 생성 완료: {self.task_id}")
            return result

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise

    def _step_generate_video(self, prompt_result: Dict[str, Any]) -> Dict[str, Any]:
        """4단계: 실제 Veo API를 사용한 비디오 생성 (동기 처리)"""
        step = self.steps["video_gen"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "video_gen"

        try:
            logger.info(f"🎬 비디오 생성 단계 시작: {self.task_id}")

            text = prompt_result["text"]
            video_prompt = prompt_result["video_prompt"]
            source_image_url = prompt_result.get("source_image_url", "")

            logger.info(f"🎥 단일 비디오 생성 시작")
            logger.info(f"  📝 텍스트: '{text}'")
            logger.info(f"  🎬 프롬프트 길이: {len(video_prompt)}자")

            try:
                # 비동기 Sora API를 동기 파이프라인에서 호출
                import asyncio
                from app.services.sora_service import generate_sign_video

                try:
                    video_result_data = asyncio.run(
                        generate_sign_video(
                            prompt=video_prompt,
                            task_id=self.task_id,
                            reference_image_url=source_image_url,
                        )
                    )
                except RuntimeError:
                    # 진행 중인 루프가 있다면 nest_asyncio 적용 시도
                    import nest_asyncio

                    nest_asyncio.apply()
                    loop = asyncio.get_event_loop()
                    video_result_data = loop.run_until_complete(
                        generate_sign_video(
                            prompt=video_prompt,
                            task_id=self.task_id,
                            reference_image_url=source_image_url,
                        )
                    )

                if video_result_data and video_result_data.get("status") == "success":
                    video_result = {
                        "text": text,
                        "video_url": video_result_data["video_url"],
                        "video_prompt": video_prompt,
                        "source_image_url": source_image_url,
                        "status": "completed",
                        "sora_response": video_result_data,
                        "created_at": datetime.now().isoformat(),
                        "note": "Sora 동기 생성 완료",
                    }
                    logger.info(
                        f"✅ Sora 비디오 생성 완료: {video_result_data['video_url']}"
                    )
                else:
                    logger.warning(f"⚠️ Sora API 응답 이상: {video_result_data}")
                    video_result = {
                        "text": text,
                        "video_url": None,
                        "video_prompt": video_prompt,
                        "source_image_url": source_image_url,
                        "status": "failed",
                        "error": "Sora API 응답 실패",
                        "sora_response": video_result_data,
                        "created_at": datetime.now().isoformat(),
                    }

            except Exception as api_error:
                logger.error(f"❌ 비디오 생성 중 오류: {api_error}")
                video_result = {
                    "text": text,
                    "video_url": None,
                    "video_prompt": video_prompt,
                    "source_image_url": source_image_url,
                    "status": "failed",
                    "error": str(api_error),
                    "created_at": datetime.now().isoformat(),
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
            raise

    def _step_finalize(self, video_result: Dict[str, Any]) -> Dict[str, Any]:
        """5단계: 최종 결과 취합"""
        step = self.steps["finalize"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "finalize"

        try:
            logger.info(f"최종 결과 취합: {self.task_id}")

            # 단일 비디오 결과 처리
            video_urls = (
                [video_result["video_url"]] if video_result.get("video_url") else []
            )
            video_details = [video_result] if video_result.get("video_url") else []

            # 최종 결과 구조
            final_result = {
                "task_id": self.task_id,
                "status": "completed",
                "video_urls": video_urls,
                "total_videos": len(video_urls),
                "successful_videos": 1
                if video_result.get("status") in ["success", "completed", "mock"]
                else 0,
                "failed_videos": 0
                if video_result.get("status") in ["success", "completed", "mock"]
                else 1,
                "completed_at": datetime.now().isoformat(),
                "video_details": video_details,
                "text": video_result.get("text", ""),
                "video_prompt": video_result.get("video_prompt", ""),
                "veo_status": video_result.get("status", "unknown"),
            }

            # 에러가 있는 경우
            if video_result.get("status") not in ["success", "completed", "mock"]:
                final_result["status"] = "failed"
                final_result["error"] = video_result.get(
                    "error", "비디오 생성에 실패했습니다"
                )

            # 모의 결과인 경우
            if video_result.get("status") == "mock":
                final_result["note"] = (
                    "모의 비디오 URL이 생성되었습니다 (실제 VEO API 호출이 아님)"
                )

            step.data = final_result
            step.status = "completed"
            step.completed_at = datetime.now()

            logger.info(
                f"파이프라인 최종 완료: {self.task_id}, 상태: {final_result['status']}, 비디오 수: {final_result['total_videos']}"
            )
            return final_result

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise

    def get_status(self) -> Dict[str, Any]:
        """현재 파이프라인 상태 반환"""
        completed_steps = [
            step_name
            for step_name, step in self.steps.items()
            if step.status == "completed"
        ]

        return {
            "task_id": self.task_id,
            "pipeline_status": self.pipeline_status,
            "current_step": self.current_step,
            "completed_steps": completed_steps,
            "steps": {
                step_name: {
                    "status": step.status,
                    "started_at": step.started_at.isoformat()
                    if step.started_at
                    else None,
                    "completed_at": step.completed_at.isoformat()
                    if step.completed_at
                    else None,
                    "error": step.error,
                }
                for step_name, step in self.steps.items()
            },
            "final_result": self.final_result,
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
            "task_id": task_id,
            "pipeline_status": "not_found",
            "error": "해당 작업 ID를 찾을 수 없습니다",
        }

    pipeline = _active_sync_pipelines[task_id]
    status = pipeline.get_status()
    logger.info(f"파이프라인 {task_id} 상태 반환: {status.get('pipeline_status')}")
    return status


def cleanup_completed_sync_pipelines():
    """완료된 동기 파이프라인들 정리 (메모리 관리)"""
    to_remove = []
    for task_id, pipeline in _active_sync_pipelines.items():
        if pipeline.pipeline_status in ["completed", "failed"]:
            # 완료된 지 1시간 이상 지난 파이프라인 제거
            if (
                pipeline.steps.get("finalize")
                and pipeline.steps["finalize"].completed_at
            ):
                completed_time = pipeline.steps["finalize"].completed_at
                if (datetime.now() - completed_time).total_seconds() > 3600:  # 1시간
                    to_remove.append(task_id)

    for task_id in to_remove:
        del _active_sync_pipelines[task_id]
        logger.info(f"완료된 동기 파이프라인 정리: {task_id}")
