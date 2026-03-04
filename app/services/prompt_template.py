import os
import re
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import jinja2

from .llm_service import get_llm_service

logger = logging.getLogger(__name__)


class PromptTemplateError(Exception):
    """프롬프트 템플릿 관련 에러 기본 클래스"""

    pass


class TemplateNotFoundError(PromptTemplateError):
    """템플릿 파일을 찾을 수 없는 에러"""

    pass


class TemplateRenderError(PromptTemplateError):
    """템플릿 렌더링 에러"""

    pass


class PromptTemplateManager:
    """프롬프트 템플릿 관리 클래스

    Jinja2를 사용하여 수어 비디오 생성용 프롬프트 템플릿을 관리합니다.
    """

    def __init__(self, templates_dir: Optional[str] = None):
        """초기화

        Args:
            templates_dir: 템플릿 디렉토리 경로 (기본값: app/templates/prompts)
        """
        if templates_dir is None:
            # 현재 파일 기준으로 템플릿 디렉토리 경로 설정
            current_dir = Path(__file__).parent.parent
            self.templates_dir = current_dir / "templates" / "prompts"
        else:
            self.templates_dir = Path(templates_dir)

        # 템플릿 디렉토리가 존재하는지 확인
        if not self.templates_dir.exists():
            raise PromptTemplateError(
                f"템플릿 디렉토리를 찾을 수 없습니다: {self.templates_dir}"
            )

        # Jinja2 환경 설정
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )

        logger.info(f"PromptTemplateManager 초기화 완료: {self.templates_dir}")

    def get_template(self, template_name: str) -> jinja2.Template:
        """템플릿 파일 로드

        Args:
            template_name: 템플릿 이름 (.txt 확장자 제외)

        Returns:
            jinja2.Template: 로드된 템플릿 객체

        Raises:
            TemplateNotFoundError: 템플릿 파일을 찾을 수 없는 경우
        """
        try:
            template_file = f"{template_name}.txt"
            template = self.env.get_template(template_file)
            logger.debug(f"템플릿 로드 성공: {template_file}")
            return template
        except jinja2.TemplateNotFound:
            raise TemplateNotFoundError(
                f"템플릿을 찾을 수 없습니다: {template_name}.txt"
            )

    def render_prompt(self, template_name: str, context: Dict[str, Any]) -> str:
        """컨텍스트 데이터로 프롬프트 생성

        Args:
            template_name: 템플릿 이름
            context: 템플릿 변수 딕셔너리

        Returns:
            str: 렌더링된 프롬프트 텍스트

        Raises:
            TemplateNotFoundError: 템플릿을 찾을 수 없는 경우
            TemplateRenderError: 템플릿 렌더링 실패
        """
        try:
            template = self.get_template(template_name)
            rendered = template.render(**context)
            logger.debug(f"프롬프트 렌더링 성공: {template_name}")
            return rendered.strip()
        except jinja2.TemplateError as e:
            raise TemplateRenderError(f"템플릿 렌더링 실패 ({template_name}): {str(e)}")

    def generate_video_prompt(
        self,
        sentence: str,
        sign_data: List[Dict[str, Any]],
        image_context: Optional[str] = None,
        reference_image_url: Optional[str] = None,
        template_name: Optional[str] = None,
    ) -> str:
        """문장과 수어 데이터로 비디오 생성 프롬프트 생성
        gemini_video_prompt.txt 템플릿을 사용하여 Gemini AI가 프롬프트를 생성합니다.

        Args:
            sentence: 변환할 문장
            sign_data: 수어 데이터 리스트
            image_context: 원본 이미지의 시각 맥락 요약 (인물/배경/조명)
            reference_image_url: 참조 이미지 URL
            template_name: 사용할 템플릿 이름 (사용되지 않음, 항상 gemini_video_prompt 사용)

        Returns:
            str: 생성된 비디오 프롬프트
        """
        try:
            # gemini_video_prompt.txt 템플릿 로드
            video_prompt_template = self._load_gemini_video_template()

            # 문장 분석 데이터 준비 (형태소 분석은 여기서는 단순화)
            sentence_analysis = f"문장: '{sentence}' (길이: {len(sentence)}자)"

            # 수어 데이터 포맷팅
            if sign_data:
                sign_data_formatted = []
                for item in sign_data:
                    if "word" in item and "description" in item:
                        sign_data_formatted.append(
                            f'- Word: "{item["word"]}" - {item["description"]}'
                        )
                sign_data_str = "\n".join(sign_data_formatted)
            else:
                sign_data_str = (
                    "No specific sign language details provided from Culture API."
                )

            # 템플릿 변수 치환
            template_with_data = video_prompt_template.replace(
                "{{sentence_analysis}}", sentence_analysis
            )
            template_with_data = template_with_data.replace(
                "{{sign_data}}", sign_data_str
            )
            template_with_data = template_with_data.replace(
                "{{image_context}}",
                (image_context or "No additional visual context provided."),
            )
            template_with_data = template_with_data.replace(
                "{{reference_image_url}}", (reference_image_url or "N/A")
            )

            # LLMService를 사용하여 프롬프트 생성
            try:
                llm = get_llm_service()

                logger.info(
                    f"🤖 LLM에게 비디오 프롬프트 생성 요청: '{sentence[:30]}...'"
                )
                generated_prompt = llm.generate_text(
                    prompt=template_with_data,
                    system_prompt="You are an expert at creating detailed video generation prompts for Korean Sign Language videos.",
                    temperature=0.7,
                )

                if generated_prompt:
                    logger.info(f"✅ LLM 프롬프트 생성 완료: {len(generated_prompt)}자")
                    return generated_prompt
                else:
                    # 빈 응답은 비정상 상황일 수 있으므로, 디버깅을 위한 LLM 컨텍스트를 함께 로그로 남긴다.
                    provider = getattr(llm, "provider", None)
                    model = getattr(llm, "model", None) or getattr(
                        llm, "model_name", None
                    )
                    llm_class = llm.__class__.__name__
                    logger.warning(
                        "⚠️ LLM 응답이 비어있음, 폴백 프롬프트 사용 (provider=%s, model=%s, llm_class=%s)",
                        provider,
                        model,
                        llm_class,
                    )
                    return self._create_fallback_prompt(
                        sentence,
                        sign_data,
                        image_context=image_context,
                        reference_image_url=reference_image_url,
                    )

            except Exception as e:
                logger.error(f"❌ LLM 프롬프트 생성 중 오류: {e}")
                return self._create_fallback_prompt(
                    sentence,
                    sign_data,
                    image_context=image_context,
                    reference_image_url=reference_image_url,
                )

        except Exception as e:
            logger.error(f"❌ 비디오 프롬프트 생성 중 오류: {e}")
            return self._create_fallback_prompt(
                sentence,
                sign_data,
                image_context=image_context,
                reference_image_url=reference_image_url,
            )

    def _load_gemini_video_template(self) -> str:
        """
        gemini_video_prompt.txt 템플릿을 로드합니다.

        Returns:
            str: 비디오 프롬프트 템플릿 내용
        """
        try:
            template_path = self.templates_dir / "gemini_video_prompt.txt"
            logger.info(f"🔍 템플릿 파일 경로: {template_path}")
            with open(template_path, "r", encoding="utf-8") as f:
                content = f.read()
                logger.info(f"✅ 템플릿 파일 로드 완료: {len(content)}자")
                return content
        except FileNotFoundError:
            logger.error("❌ gemini_video_prompt.txt 템플릿 파일을 찾을 수 없습니다.")
            raise
        except Exception as e:
            logger.error(f"❌ 비디오 프롬프트 템플릿 로드 실패: {e}")
            raise

    def _create_fallback_prompt(
        self,
        sentence: str,
        sign_data: List[Dict[str, Any]],
        image_context: Optional[str] = None,
        reference_image_url: Optional[str] = None,
    ) -> str:
        """
        폴백용 기본 프롬프트를 생성합니다. (Sora 2 포맷)

        Args:
            sentence: 문장
            sign_data: 수어 데이터

        Returns:
            str: 폴백 프롬프트
        """
        prompt = f"""Character performing Korean Sign Language (KSL) based on the reference image. The clothing, background, and lighting are clear and well-lit.

Camera and Style:
- Shot: Medium close-up, eye-level, focusing on the upper body and hands
- Vibe: Professional, clean studio, educational

Action:
- The character begins performing Korean Sign Language fluidly and naturally to convey the following meaning: "{sentence}" """

        # 수어 데이터 추가
        if sign_data:
            prompt += "\n- Specific sign language movements:"
            for item in sign_data:
                if "word" in item and "description" in item:
                    prompt += f"\n  * {item['word']}: {item['description']}"
        else:
            prompt += "\n- The character creatively signs the story using appropriate interpretative gestures."

        prompt += f"""
- Ensure the character maintains eye contact and transitions smoothly between signs without any text or subtitles appearing anywhere.

Dialogue:
- Character: "{sentence}"

Ambient Sound: Quiet room tone

Reference Image URL: {reference_image_url or 'N/A'}

Visual Context from Reference Image:
{image_context or 'No additional visual context provided.'}
"""
        return prompt

    def _select_template_for_sentence(self, sentence: str) -> str:
        """문장 유형에 따라 적절한 템플릿 선택

        Args:
            sentence: 분석할 문장

        Returns:
            str: 선택된 템플릿 이름
        """
        sentence = sentence.strip()

        # 질문 문장 감지
        if self._is_question(sentence):
            return "question_prompt"

        # 명령 문장 감지
        if self._is_command(sentence):
            return "command_prompt"

        # 기본 평서문
        return "statement_prompt"

    def _is_question(self, sentence: str) -> bool:
        """질문 문장 여부 확인

        Args:
            sentence: 확인할 문장

        Returns:
            bool: 질문 문장 여부
        """
        # 물음표로 끝나는 경우
        if sentence.endswith("?") or sentence.endswith("？"):
            return True

        # 의문사가 포함된 경우
        question_words = [
            "무엇",
            "뭐",
            "언제",
            "어디",
            "누구",
            "왜",
            "어떻게",
            "어떤",
            "몇",
            "얼마",
            "어느",
            "어디서",
            "누가",
            "언제부터",
            "언제까지",
        ]

        return any(q_word in sentence for q_word in question_words)

    def _is_command(self, sentence: str) -> bool:
        """명령 문장 여부 확인

        Args:
            sentence: 확인할 문장

        Returns:
            bool: 명령 문장 여부
        """
        # 명령형 어미 패턴
        command_patterns = [
            r".*라$",  # ~라
            r".*해라$",  # ~해라
            r".*하세요$",  # ~하세요
            r".*가세요$",  # ~가세요
            r".*하십시오$",  # ~하십시오
            r".*해$",  # ~해
            r".*하자$",  # ~하자
            r".*해줘$",  # ~해줘
            r".*가자$",  # ~가자
            r".*와$",  # ~와
            r".*오세요$",  # ~오세요
            r".*읽어라$",  # ~읽어라
            r".*보세요$",  # ~보세요
            r".*드세요$",  # ~드세요
        ]

        return any(re.match(pattern, sentence) for pattern in command_patterns)

    def list_available_templates(self) -> List[str]:
        """사용 가능한 템플릿 목록 반환

        Returns:
            List[str]: 템플릿 이름 리스트 (.txt 확장자 제외)
        """
        templates = []
        for file_path in self.templates_dir.glob("*.txt"):
            templates.append(file_path.stem)

        logger.debug(f"사용 가능한 템플릿: {templates}")
        return sorted(templates)

    def validate_template(self, template_name: str) -> bool:
        """템플릿 파일 유효성 확인

        Args:
            template_name: 확인할 템플릿 이름

        Returns:
            bool: 템플릿 유효성 여부
        """
        try:
            self.get_template(template_name)
            return True
        except TemplateNotFoundError:
            return False


# 전역 인스턴스 (선택적)
_default_manager: Optional[PromptTemplateManager] = None


def get_default_prompt_manager() -> PromptTemplateManager:
    """기본 프롬프트 템플릿 매니저 인스턴스 반환

    Returns:
        PromptTemplateManager: 기본 매니저 인스턴스
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = PromptTemplateManager()
    return _default_manager
