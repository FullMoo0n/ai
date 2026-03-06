"""
통합 파이프라인 서비스

S3 이미지 URL에서 수어 비디오 생성까지의 전체 파이프라인을 구현합니다.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import uuid
import sys
import os
from datetime import datetime

from app.services.vision_s3 import process_s3_image_with_vision
from app.services.openai_vision import analyze_image_context_with_openai
from app.services.sentence_segmenter import split_sentences
from app.services.tokenizer import tokenize
from app.services.sign_data_service import SignDataService
from .prompt_template import get_default_prompt_manager

logger = logging.getLogger(__name__)

# Initialize Sign Service
_sign_data_service = SignDataService()


class PipelineError(Exception):
    """파이프라인 처리 중 발생하는 에러"""

    pass


class PipelineStep:
    """파이프라인 단계별 결과를 담는 클래스"""

    def __init__(self, step_name: str, status: str = "pending"):
        self.step_name = step_name
        self.status = status  # pending, processing, completed, failed
        self.data = None
        self.error = None
        self.started_at = None
        self.completed_at = None


class IntegratedPipeline:
    """통합 파이프라인 클래스"""

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
            "sentence_processing",  # 통합 토큰 처리 (모든 토큰 합쳐서 처리)
            "video_gen",  # 단일 통합 비디오 생성
            "finalize",  # 최종 결과 취합
        ]

        for step_name in step_names:
            self.steps[step_name] = PipelineStep(step_name)

    async def execute(self, s3_image_url: str) -> Dict[str, Any]:
        """전체 파이프라인 실행

        Args:
            s3_image_url: S3 이미지 URL

        Returns:
            Dict: 최종 결과
        """
        try:
            self.pipeline_status = "processing"
            self.source_image_url = s3_image_url
            logger.info(f"파이프라인 시작: {self.task_id}")

            # 1. OCR 텍스트 추출
            extracted_text = await self._step_ocr(s3_image_url)

            # 2. 문장 분할
            sentences = await self._step_segmentation(extracted_text)

            # 3. 통합 토큰 처리 (모든 토큰 합쳐서 하나의 프롬프트 생성)
            integrated_result = await self._step_process_sentences(sentences)

            # 4. 단일 비디오 생성
            video_result = await self._step_generate_videos([integrated_result])

            # 5. 최종 결과 취합
            final_result = await self._step_finalize(video_result)

            self.pipeline_status = "completed"
            self.final_result = final_result

            logger.info(f"파이프라인 완료: {self.task_id}")
            return final_result

        except Exception as e:
            self.pipeline_status = "failed"
            logger.error(f"파이프라인 실패: {self.task_id}, 오류: {e}")
            raise PipelineError(f"파이프라인 실행 실패: {str(e)}")

    async def _step_ocr(self, s3_image_url: str) -> str:
        """1단계: OCR 텍스트 추출"""
        step = self.steps["ocr"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "ocr"

        try:
            logger.info(f"OCR 단계 시작: {self.task_id}")

            result = await process_s3_image_with_vision(
                s3_url=s3_image_url,
                feature="TEXT_DETECTION",
                language_hints=["ko"],
                include_word_boxes=False,
            )

            extracted_text = result.get("text", "")

            try:
                context_result = await analyze_image_context_with_openai(s3_image_url)
                self.image_context = context_result.get("image_context", "")
                self.character_description = context_result.get(
                    "character_description", ""
                )
            except Exception as context_error:
                logger.warning(
                    f"이미지 맥락/캐릭터 분석 실패 (OCR은 계속 진행): {context_error}"
                )
                self.image_context = ""
                self.character_description = ""

            if not extracted_text.strip():
                raise PipelineError("이미지에서 텍스트를 추출할 수 없습니다")

            step.data = extracted_text
            step.status = "completed"
            step.completed_at = datetime.now()

            logger.info(f"OCR 완료: {self.task_id}, 텍스트 길이: {len(extracted_text)}")
            if self.image_context:
                logger.info(
                    f"🖼️ 이미지 맥락 요약 길이: {len(self.image_context)}"
                )
            if self.character_description:
                logger.info(
                    f"🧸 캐릭터 설명 추출 완료: {self.character_description[:80]}"
                )
            return extracted_text

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise

    async def _step_segmentation(self, text: str) -> List[str]:
        """2단계: 문장 분할"""
        step = self.steps["segmentation"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "segmentation"

        try:
            logger.info(f"문장 분할 단계 시작: {self.task_id}")

            sentences = split_sentences(text)

            # 빈 문장 제거
            sentences = [s.strip() for s in sentences if s.strip()]

            if not sentences:
                raise PipelineError("분할할 수 있는 문장이 없습니다")

            step.data = sentences
            step.status = "completed"
            step.completed_at = datetime.now()

            logger.info(f"문장 분할 완료: {self.task_id}, 문장 수: {len(sentences)}")
            return sentences

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            raise

    async def _step_process_sentences(self, sentences: List[str]) -> Dict[str, Any]:
        """3단계: 문장별 Qdrant 벡터 검색 및 프롬프트 생성 -> 통합 토큰 처리로 변경"""
        step = self.steps["sentence_processing"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "sentence_processing"

        try:
            logger.info(f"통합 토큰 처리 단계 시작: {self.task_id}")

            # 모든 문장에서 토큰 수집
            all_tokens = []
            all_sentences = sentences

            for sentence in sentences:
                # 각 문장을 토큰화
                tokens = tokenize(sentence)
                all_tokens.extend(tokens)

            # 중복 토큰 제거 (순서 유지)
            unique_tokens = []
            seen = set()
            for token in all_tokens:
                if token not in seen and len(token.strip()) > 0:
                    unique_tokens.append(token)
                    seen.add(token)

            logger.info(f"📚 전체 문장 수: {len(all_sentences)}")
            logger.info(f"📝 전체 토큰 수: {len(all_tokens)}")
            logger.info(f"🔍 고유 토큰 수: {len(unique_tokens)}")

            # Qdrant 벡터 검색
            sign_data = []
            api_total_count = 0
            api_success_count = 0

            logger.info(f"🌐 Qdrant 벡터 검색 시작: {len(unique_tokens)}개 토큰")

            # 의미 있는 토큰만 필터링
            meaningful_tokens = [
                token
                for token in unique_tokens
                if len(token.strip()) > 1
                and token not in ["·", "!", "?", ".", ",", "'", '"']
            ]
            skipped_count = len(unique_tokens) - len(meaningful_tokens)
            if skipped_count:
                logger.info(f"  ⏭️ 의미 없는 토큰 {skipped_count}개 건너뜀")

            api_total_count = len(meaningful_tokens)

            # 동시성 제어용 세마포어 (Copilot Review #3 반영)
            sem = asyncio.Semaphore(5)

            async def _search_token(token: str) -> dict | None:
                """개별 토큰을 비동기로 검색 (세마포어 적용)"""
                async with sem:
                    try:
                        loop = asyncio.get_running_loop()
                        description = await loop.run_in_executor(
                            None, _sign_data_service.search_sign_description, token
                        )
                        if description and description.strip():
                            logger.info(f"    ✅ 검색 성공: '{token}'")
                            logger.info(f"    📝 수어 설명 전문: {description}")
                            return {
                                "word": token,
                                "description": description,
                                "culture_data": {"description": description},
                            }
                        else:
                            logger.info(
                                f"    ❌ 데이터 없음: '{token}' (프롬프트에서 제외)"
                            )
                            return None
                    except Exception as e:
                        logger.warning(
                            f"    🚨 검색 오류: '{token}' - {str(e)} (프롬프트에서 제외)"
                        )
                        return None

            # 병렬 검색 실행
            results = await asyncio.gather(
                *[_search_token(t) for t in meaningful_tokens]
            )
            sign_data = [r for r in results if r is not None]
            api_success_count = len(sign_data)

            logger.info(
                f"📊 Qdrant 검색 결과: {api_success_count}/{api_total_count} 성공"
            )
            logger.info(f"📚 프롬프트 포함 수어 데이터: {len(sign_data)}개")

            # 통합 프롬프트 생성
            full_text = " ".join(all_sentences)
            gloss_sequence = " ".join(meaningful_tokens) if meaningful_tokens else full_text
            prompt_manager = get_default_prompt_manager()
            video_prompt = prompt_manager.build_video_prompt(
                original_text=full_text,
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
            logger.info(f"📝 통합 프롬프트 생성 완료: {len(video_prompt)}자")

            # 결과 구조 (단일 결과로 변경)
            integrated_result = {
                "full_text": full_text,
                "sentences": all_sentences,
                "total_tokens": all_tokens,
                "unique_tokens": unique_tokens,
                "sign_data": sign_data,  # 실제 API 데이터만 포함
                "image_context": self.image_context,
                "character_description": self.character_description,
                "gloss_sequence": gloss_sequence,
                "source_image_url": self.source_image_url,
                "video_prompt": video_prompt,
                "api_stats": {
                    "total_unique_tokens": len(unique_tokens),
                    "processed_tokens": api_total_count,
                    "successful_api_calls": api_success_count,
                    "included_in_prompt": len(sign_data),
                },
            }

            step.data = integrated_result
            step.status = "completed"
            step.completed_at = datetime.now()

            logger.info(f"✅ 통합 토큰 처리 완료")
            logger.info(f"   📊 총 API 호출: {api_success_count}/{api_total_count}")
            logger.info(f"   📝 프롬프트 포함 수어 데이터: {len(sign_data)}개")

            return integrated_result

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            logger.error(f"❌ 통합 토큰 처리 실패: {e}")
            raise

    async def _step_generate_videos(
        self, integrated_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """4단계: 단일 통합 비디오 생성"""
        step = self.steps["video_gen"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "video_gen"

        try:
            # 통합 결과에서 첫 번째 (유일한) 결과 사용
            integrated_result = integrated_results[0] if integrated_results else {}

            logger.info(f"통합 비디오 생성 단계 시작: {self.task_id}")

            video_results = []

            # 통합 프롬프트로 단일 비디오 생성
            full_text = integrated_result.get("full_text", "")
            video_prompt = integrated_result.get("video_prompt", "")
            source_image_url = integrated_result.get("source_image_url", "")

            try:
                # Sora API 호출
                from .sora_service import generate_sign_video

                video_result_data = await generate_sign_video(
                    prompt=video_prompt,
                    task_id=self.task_id,
                    reference_image_url=source_image_url,
                )

                if video_result_data and video_result_data.get("status") == "success":
                    video_result = {
                        "full_text": full_text,
                        "video_url": video_result_data["video_url"],
                        "video_prompt": video_prompt,
                        "source_image_url": source_image_url,
                        "status": "completed",
                        "sora_response": video_result_data,
                        "note": video_result_data.get("note", ""),
                        "integrated_data": integrated_result,
                    }
                    logger.info(
                        f"✅ 통합 비디오 생성 성공: {video_result_data['video_url']}"
                    )
                else:
                    # Sora API 응답이 예상과 다른 경우
                    logger.warning(
                        f"⚠️ Sora API 응답이 예상과 다름: {video_result_data}"
                    )

                    video_result = {
                        "full_text": full_text,
                        "video_url": None,
                        "video_prompt": video_prompt,
                        "source_image_url": source_image_url,
                        "status": "failed",
                        "sora_response": video_result_data,
                        "note": "Sora API 응답이 예상과 다름",
                        "integrated_data": integrated_result,
                    }

            except Exception as e:
                logger.error(f"❌ 통합 비디오 생성 실패: {e}")

                video_result = {
                    "full_text": full_text,
                    "video_url": None,
                    "video_prompt": video_prompt,
                    "source_image_url": source_image_url,
                    "status": "failed",
                    "sora_response": None,
                    "note": f"오류: {str(e)}",
                    "integrated_data": integrated_result,
                }
                logger.info(f"🚨 오류 비디오 처리 결과 생성: {str(e)}")

            video_results.append(video_result)

            step.data = video_results
            step.status = "completed"
            step.completed_at = datetime.now()

            logger.info(f"✅ 통합 비디오 생성 완료: {self.task_id}")
            return video_results

        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            step.completed_at = datetime.now()
            logger.error(f"❌ 통합 비디오 생성 단계 실패: {e}")
            raise

    async def _step_finalize(
        self, video_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """5단계: 최종 결과 취합"""
        step = self.steps["finalize"]
        step.status = "processing"
        step.started_at = datetime.now()
        self.current_step = "finalize"

        try:
            logger.info(f"최종 결과 취합: {self.task_id}")

            # 성공한 비디오들만 필터링
            successful_videos = [
                result
                for result in video_results
                if result.get("status") == "completed" and "video_url" in result
            ]

            failed_videos = [
                result
                for result in video_results
                if result.get("status") == "failed" or "error" in result
            ]

            video_urls = [video["video_url"] for video in successful_videos]

            final_result = {
                "task_id": self.task_id,
                "status": "completed",
                "video_urls": video_urls,
                "total_videos": len(video_results),
                "successful_videos": len(successful_videos),
                "failed_videos": len(failed_videos),
                "completed_at": datetime.now().isoformat(),
                "video_details": successful_videos,
            }

            if failed_videos:
                final_result["failed_details"] = failed_videos
                if len(failed_videos) == len(video_results):
                    final_result["status"] = "failed"
                    final_result["error"] = "모든 비디오 생성이 실패했습니다"

            step.data = final_result
            step.status = "completed"
            step.completed_at = datetime.now()

            logger.info(
                f"파이프라인 최종 완료: {self.task_id}, 성공 비디오: {len(successful_videos)}"
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
_active_pipelines: Dict[str, IntegratedPipeline] = {}


async def start_pipeline(s3_image_url: str, task_id: str) -> str:
    """파이프라인 시작

    Args:
        s3_image_url: S3 이미지 URL
        task_id: 작업 ID

    Returns:
        str: 작업 ID
    """
    pipeline = IntegratedPipeline(task_id)
    _active_pipelines[task_id] = pipeline

    # 백그라운드에서 파이프라인 실행
    asyncio.create_task(pipeline.execute(s3_image_url))

    return task_id


def get_pipeline_status(task_id: str) -> Dict[str, Any]:
    """파이프라인 상태 조회

    Args:
        task_id: 작업 ID

    Returns:
        Dict: 파이프라인 상태
    """
    if task_id not in _active_pipelines:
        return {
            "task_id": task_id,
            "pipeline_status": "not_found",
            "error": "해당 작업 ID를 찾을 수 없습니다",
        }

    pipeline = _active_pipelines[task_id]
    return pipeline.get_status()


def cleanup_completed_pipelines():
    """완료된 파이프라인들 정리 (메모리 관리)"""
    to_remove = []
    for task_id, pipeline in _active_pipelines.items():
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
        del _active_pipelines[task_id]
        logger.info(f"완료된 파이프라인 정리: {task_id}")
