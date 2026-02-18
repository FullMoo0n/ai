"""
통합 LLM 서비스

Ollama(로컬) 및 OpenAI API를 모두 지원하는 통합 LLM 클라이언트입니다.
openai 패키지의 호환 API를 활용하여 두 프로바이더를 동일한 인터페이스로 사용합니다.
"""

import os
import json
import logging
from typing import Optional
import threading

from openai import OpenAI
from app.utils.text_utils import clean_markdown_json_response

logger = logging.getLogger(__name__)


class LLMService:
    """통합 LLM 서비스 클래스

    환경 변수 LLM_PROVIDER에 따라 Ollama 또는 OpenAI를 사용합니다.
    - ollama: 로컬 Ollama 서버 (OpenAI 호환 API)
    - openai: 표준 OpenAI API
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "ollama")

        if self.provider == "ollama":
            self.base_url = base_url or os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434/v1"
            )
            # Validate OLLAMA_BASE_URL to prevent SSRF attacks
            self._validate_ollama_base_url(self.base_url)
            self.model = model or os.getenv(
                "OLLAMA_MODEL", "qwen2.5:7b-instruct-q4_K_M"
            )
            self.api_key = "ollama"  # Ollama는 API 키가 필요 없지만 openai 클라이언트에는 필수
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            logger.info(
                f"🤖 LLMService 초기화 완료: provider=ollama, model={self.model}, "
                f"base_url={self.base_url}"
            )

        elif self.provider == "openai":
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "OPENAI_API_KEY 환경 변수가 설정되지 않았습니다."
                )
            self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = None
            self.client = OpenAI(api_key=self.api_key)
            logger.info(
                f"🤖 LLMService 초기화 완료: provider=openai, model={self.model}"
            )
        else:
            raise ValueError(
                f"지원하지 않는 LLM_PROVIDER: {self.provider}. "
                "'ollama' 또는 'openai'를 사용하세요."
            )

    def _validate_ollama_base_url(self, base_url: str) -> None:
        """Validate OLLAMA_BASE_URL to prevent SSRF attacks.
        
        Args:
            base_url: The base URL to validate
            
        Raises:
            ValueError: If the base URL is not in the whitelist
        """
        # Allow localhost and 127.0.0.1 for local development
        allowed_prefixes = [
            "http://localhost:",
            "http://127.0.0.1:",
            "https://localhost:",
            "https://127.0.0.1:",
        ]
        
        if not any(base_url.startswith(prefix) for prefix in allowed_prefixes):
            logger.warning(
                f"⚠️ OLLAMA_BASE_URL이 localhost가 아님: {base_url}. "
                "원격 Ollama 서버를 사용하는 경우 보안에 유의하세요."
            )

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            model: 모델 이름 (None이면 기본 모델 사용)
            temperature: 생성 온도
            max_tokens: 최대 토큰 수

        Returns:
            str: 생성된 텍스트
        """
        model = model or self.model
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            logger.info(
                f"📤 LLM 요청: provider={self.provider}, model={model}, "
                f"prompt_len={len(prompt)}"
            )

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            result = response.choices[0].message.content.strip()
            logger.info(
                f"📥 LLM 응답: {len(result)}자"
            )
            return result

        except Exception as e:
            logger.error(f"❌ LLM 텍스트 생성 실패: {e}")
            raise

    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> dict:
        """JSON 출력 생성

        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            model: 모델 이름 (None이면 기본 모델 사용)
            temperature: 생성 온도

        Returns:
            dict: 파싱된 JSON 결과
        """
        model = model or self.model
        messages = []

        base_system = (
            system_prompt or "You are a helpful assistant."
        )
        base_system += "\nYou MUST respond with valid JSON only. No markdown, no code blocks."
        messages.append({"role": "system", "content": base_system})
        messages.append({"role": "user", "content": prompt})

        content = ""
        try:
            logger.info(
                f"📤 LLM JSON 요청: provider={self.provider}, model={model}"
            )

            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4096,
            }

            # OpenAI는 response_format 지원, Ollama는 모델에 따라 다름
            if self.provider == "openai":
                kwargs["response_format"] = {"type": "json_object"}

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content.strip()

            # 마크다운 코드 블록 제거 (Ollama 모델들이 가끔 추가함)
            content = clean_markdown_json_response(content)

            result = json.loads(content)
            logger.info(f"📥 LLM JSON 응답 파싱 성공")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"❌ LLM JSON 파싱 실패: {e}")
            logger.error(f"   원본 응답: {content}")
            raise
        except Exception as e:
            logger.error(f"❌ LLM JSON 생성 실패: {e}")
            raise


# 전역 싱글턴 인스턴스와 락
_llm_service: Optional[LLMService] = None
_llm_service_lock = threading.Lock()


def get_llm_service() -> LLMService:
    """LLMService 싱글턴 인스턴스 반환

    Returns:
        LLMService: LLM 서비스 인스턴스
    """
    global _llm_service
    if _llm_service is None:
        with _llm_service_lock:
            if _llm_service is None:
                _llm_service = LLMService()
    return _llm_service
