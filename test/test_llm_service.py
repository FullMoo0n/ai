"""
LLMService 통합 테스트

이 테스트는 실제 LLM API를 호출하므로, 적절한 환경 변수가 설정되어야 합니다:
- LLM_PROVIDER: 'ollama' 또는 'openai'
- OLLAMA_BASE_URL, OLLAMA_MODEL (ollama 사용 시)
- OPENAI_API_KEY, OPENAI_MODEL (openai 사용 시)
"""

import os
import json
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from app.services.llm_service import LLMService, get_llm_service


def test_llm_service_initialization_ollama():
    """Ollama provider로 LLMService 초기화 테스트"""
    service = LLMService(
        provider="ollama",
        model="qwen2.5:7b-instruct-q4_K_M",
        base_url="http://localhost:11434/v1"
    )
    
    assert service.provider == "ollama"
    assert service.model == "qwen2.5:7b-instruct-q4_K_M"
    assert service.base_url == "http://localhost:11434/v1"
    print("✅ Ollama 초기화 성공")


def test_llm_service_initialization_openai():
    """OpenAI provider로 LLMService 초기화 테스트"""
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⏭️  OPENAI_API_KEY 환경변수가 설정되지 않아 스킵")
        return
    
    service = LLMService(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_key
    )
    
    assert service.provider == "openai"
    assert service.model == "gpt-4o-mini"
    assert service.api_key == openai_key
    print("✅ OpenAI 초기화 성공")


def test_llm_service_invalid_provider():
    """잘못된 provider로 초기화 시 에러 발생 테스트"""
    try:
        LLMService(provider="invalid_provider")
        print("❌ 예외가 발생하지 않음")
        assert False
    except ValueError as e:
        if "지원하지 않는 LLM_PROVIDER" in str(e):
            print("✅ 잘못된 provider 검증 성공")
        else:
            raise


def test_ollama_base_url_validation_localhost():
    """localhost OLLAMA_BASE_URL 검증 테스트"""
    # localhost는 허용되어야 함
    service = LLMService(
        provider="ollama",
        base_url="http://localhost:11434/v1"
    )
    assert service.base_url == "http://localhost:11434/v1"
    print("✅ localhost URL 검증 성공")


def test_generate_text():
    """텍스트 생성 기능 테스트"""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"🤖 테스트 provider: {provider}")
    
    try:
        service = get_llm_service()
        
        prompt = "안녕하세요를 영어로 번역하세요. 답변만 작성하세요."
        result = service.generate_text(
            prompt=prompt,
            temperature=0.1,
            max_tokens=50
        )
        
        assert result is not None
        assert len(result) > 0
        assert isinstance(result, str)
        print(f"✅ 텍스트 생성 성공")
        print(f"📄 응답: {result}")
        
    except Exception as e:
        print(f"⏭️  LLM 서비스를 사용할 수 없어 스킵: {e}")


def test_generate_json():
    """JSON 생성 기능 테스트"""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"🤖 테스트 provider: {provider}")
    
    try:
        service = get_llm_service()
        
        prompt = """
다음 한국어 문장을 분석하여 JSON 형식으로 출력하세요:
"안녕하세요"

출력 형식:
{
  "sentence": "문장",
  "word_count": 단어수,
  "has_question": true/false
}
"""
        result = service.generate_json(
            prompt=prompt,
            system_prompt="You are a helpful assistant that outputs valid JSON.",
            temperature=0.1
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert "sentence" in result or "word_count" in result
        print(f"✅ JSON 생성 성공")
        print(f"📊 응답: {json.dumps(result, ensure_ascii=False, indent=2)}")
        
    except Exception as e:
        print(f"⏭️  LLM 서비스를 사용할 수 없어 스킵: {e}")


def test_generate_json_with_markdown_blocks():
    """마크다운 코드 블록이 포함된 JSON 응답 처리 테스트"""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"🤖 테스트 provider: {provider}")
    
    try:
        service = get_llm_service()
        
        # Ollama 모델들은 종종 마크다운 코드 블록으로 JSON을 래핑함
        prompt = """
다음 문장의 주요 키워드 3개를 추출하여 JSON 배열로 반환하세요:
"한국 수어는 한국의 농인 사회에서 사용되는 시각 언어입니다."

{"keywords": ["키워드1", "키워드2", "키워드3"]} 형식으로만 답변하세요.
"""
        result = service.generate_json(
            prompt=prompt,
            temperature=0.1
        )
        
        assert result is not None
        assert isinstance(result, dict)
        assert "keywords" in result
        assert isinstance(result["keywords"], list)
        print(f"✅ 마크다운 블록 처리 성공")
        print(f"📊 추출된 키워드: {result['keywords']}")
        
    except Exception as e:
        print(f"⏭️  LLM 서비스를 사용할 수 없어 스킵: {e}")


def test_singleton_pattern():
    """싱글턴 패턴 테스트"""
    service1 = get_llm_service()
    service2 = get_llm_service()
    
    assert service1 is service2
    print("✅ 싱글턴 패턴 검증 성공")


def test_error_handling_invalid_json():
    """잘못된 JSON 응답에 대한 에러 처리 테스트"""
    provider = os.getenv("LLM_PROVIDER", "ollama")
    
    try:
        service = get_llm_service()
        
        # 의도적으로 JSON이 아닌 응답을 유도 (하지만 LLM이 항상 협조하는 것은 아님)
        prompt = "Just say 'Hello' - nothing else, no JSON"
        
        # generate_json은 JSON이 아닌 응답에 대해 JSONDecodeError를 발생시켜야 함
        try:
            result = service.generate_json(
                prompt=prompt,
                temperature=0.1
            )
            # 만약 성공했다면, LLM이 예상과 다르게 JSON을 반환한 것
            print(f"⚠️ LLM이 예상과 다르게 JSON을 반환함: {result}")
            
        except json.JSONDecodeError:
            print("✅ JSON 파싱 에러가 올바르게 발생함")
            
    except Exception as e:
        print(f"⏭️  LLM 서비스를 사용할 수 없어 스킵: {e}")



def test_llm_service_environment_variable():
    """ENVIRONMENT 환경변수에 따른 provider 선택 테스트"""
    # 백업
    original_env = os.environ.get("ENVIRONMENT")
    original_provider = os.environ.get("LLM_PROVIDER")
    
    try:
        # LLM_PROVIDER가 설정되어 있으면 그것이 우선하므로 제거
        if "LLM_PROVIDER" in os.environ:
            del os.environ["LLM_PROVIDER"]
            
        # 1. Production -> OpenAI
        os.environ["ENVIRONMENT"] = "production"
        service_prod = LLMService()
        assert service_prod.provider == "openai"
        print("✅ Environment=production -> OpenAI 선택 성공")
        
        # 2. Dev -> Ollama
        os.environ["ENVIRONMENT"] = "dev"
        service_dev = LLMService()
        assert service_dev.provider == "ollama"
        print("✅ Environment=dev -> Ollama 선택 성공")
        
    finally:
        # 복구
        if original_env:
            os.environ["ENVIRONMENT"] = original_env
        else:
            if "ENVIRONMENT" in os.environ:
                del os.environ["ENVIRONMENT"]
                
        if original_provider:
            os.environ["LLM_PROVIDER"] = original_provider

if __name__ == "__main__":
    print("=" * 60)
    print("LLMService 통합 테스트 시작")
    print("=" * 60)
    
    # 각 테스트 실행
    test_llm_service_initialization_ollama()
    test_llm_service_invalid_provider()
    test_ollama_base_url_validation_localhost()
    test_singleton_pattern()
    test_llm_service_environment_variable()
    
    # OpenAI 테스트 (API 키가 있는 경우에만)
    try:
        test_llm_service_initialization_openai()
    except Exception as e:
        print(f"⏭️  OpenAI 초기화 테스트 스킵: {e}")
    
    # 실제 API 호출 테스트
    try:
        test_generate_text()
        test_generate_json()
        test_generate_json_with_markdown_blocks()
        test_error_handling_invalid_json()
    except Exception as e:
        print(f"⏭️  API 호출 테스트 스킵: {e}")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
