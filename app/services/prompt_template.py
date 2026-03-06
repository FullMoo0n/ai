import os
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
import jinja2

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

    def build_video_prompt(
        self,
        original_text: str,
        gloss_sequence: str,
        sign_data: List[Dict[str, Any]],
        image_context: str,
        character_description: str,
        book_id: str,
        page_number: int,
        sentence_idx: int,
        reference_image_url: Optional[str] = None,
        duration_sec: int = 8,
        version: str = "v1",
    ) -> str:
        """데모용 Sora 비디오 프롬프트를 템플릿에 즉시 렌더링하여 생성합니다.

        LLM 재작성 단계를 거치지 않고 템플릿 값을 직접 치환해
        일관된 출력 포맷을 보장합니다.
        """
        try:
            video_prompt_template = self._load_gemini_video_template()
            sign_data_str = self._format_sign_data(sign_data)

            replacements = {
                "original_text": original_text,
                "gloss_sequence": gloss_sequence,
                "sign_data": sign_data_str,
                "image_context": self._normalize_multiline_text(image_context)
                or "No additional visual context provided.",
                "character_description": character_description
                or "storybook illustrated character",
                "duration_sec": str(duration_sec),
                "book_id": book_id,
                "page_number": str(page_number),
                "sentence_idx": str(sentence_idx),
                "version": version,
                "reference_image_url": reference_image_url or "N/A",
            }

            rendered = video_prompt_template
            for key, value in replacements.items():
                rendered = self._replace_template_var(rendered, key, value)

            return rendered.strip()

        except Exception as e:
            logger.error(f"❌ 비디오 프롬프트 생성 중 오류: {e}")
            return self._create_fallback_prompt(
                original_text=original_text,
                gloss_sequence=gloss_sequence,
                sign_data=sign_data,
                image_context=image_context,
                character_description=character_description,
                book_id=book_id,
                page_number=page_number,
                sentence_idx=sentence_idx,
                reference_image_url=reference_image_url,
                duration_sec=duration_sec,
                version=version,
            )

    def _replace_template_var(self, content: str, key: str, value: str) -> str:
        """템플릿 변수 치환: {{key}} 및 {key} 형식을 모두 지원."""
        return content.replace(f"{{{{{key}}}}}", value).replace(f"{{{key}}}", value)

    def _format_sign_data(self, sign_data: List[Dict[str, Any]]) -> str:
        """RAG 결과를 프롬프트 텍스트 블록으로 변환"""
        if not sign_data:
            return "- No specific sign language details provided from RAG."

        lines = []
        for item in sign_data:
            word = item.get("word")
            description = item.get("description")
            if word and description:
                lines.append(f"- {word}: {description}")

        if not lines:
            return "- No valid sign descriptions found from RAG."
        return "\n".join(lines)

    def _normalize_multiline_text(self, value: str) -> str:
        """멀티라인 텍스트의 들여쓰기/공백을 정규화"""
        if not value:
            return ""

        normalized_newline = value.replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in normalized_newline.split("\n")]

        compact_lines: List[str] = []
        previous_blank = False

        for line in lines:
            if not line:
                if not previous_blank:
                    compact_lines.append("")
                previous_blank = True
                continue

            compact_lines.append(line)
            previous_blank = False

        compact_text = "\n".join(compact_lines).strip()
        return re.sub(r"[ \t]+", " ", compact_text)

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
        original_text: str,
        gloss_sequence: str,
        sign_data: List[Dict[str, Any]],
        image_context: str,
        character_description: str,
        book_id: str,
        page_number: int,
        sentence_idx: int,
        reference_image_url: Optional[str] = None,
        duration_sec: int = 8,
        version: str = "v1",
    ) -> str:
        """템플릿 렌더링 실패 시 데모용 폴백 프롬프트"""
        sign_data_str = self._format_sign_data(sign_data)
        normalized_context = self._normalize_multiline_text(image_context)
        return f"""Generate a Korean Sign Language (KSL) video scene.

    Character:
    - {character_description or 'storybook illustrated character'}
    - Keep the same illustrated character design consistently throughout the video.

    Scene:
    - {normalized_context or 'No additional visual context provided.'}
    - Match the original storybook illustration style and mood.

    Signing content:
    - Original text: {original_text}
    - KSL gloss sequence (strict order): {gloss_sequence}
    - Sign references:
    {sign_data_str}

    Performance rules:
    - Use natural KSL facial expressions and body language.
    - Signing speed: slow and clear for children.
    - Camera: fixed front-facing, upper body visible, both hands always in frame.

    Video output:
    - Duration: {duration_sec} seconds
    - Resolution: 1280x720
    - Smooth sign transitions, no abrupt pauses.
    - No subtitles, captions, or any text overlay.
    - Keep cartoon/illustration visual style (not photorealistic)."""

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
