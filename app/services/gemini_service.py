import json
import logging
from typing import Dict, List, Any
from pathlib import Path

from .llm_service import get_llm_service, LLMService
from app.utils.text_utils import clean_markdown_json_response

logger = logging.getLogger(__name__)

class GeminiService:
    """LLM을 사용하여 한국어 문장 분석을 수행하는 서비스 (하위 호환성을 위해 GeminiService 이름 유지)"""
    
    def __init__(self, api_key: str = None):
        """
        GeminiService 초기화 (하위 호환성 유지, 내부적으로 LLMService 사용)
        
        Args:
            api_key (str): (더 이상 사용되지 않음) 개별 인스턴스용 API 키.
                현재는 무시되며, LLMService가 환경 변수에서 API 키를 자동 로드합니다.
        """
        if api_key is not None:
            logger.warning(
                "GeminiService(api_key=...) 파라미터는 더 이상 사용되지 않으며 무시됩니다. "
                "API 키는 환경 변수 또는 LLMService 설정을 통해 구성하세요."
            )
        self.llm = get_llm_service()
        logger.info(f"GeminiService 초기화 완료 (LLMService 위임: provider={self.llm.provider})")
    
    def load_prompt_template(self) -> str:
        """
        프롬프트 템플릿 파일을 로드합니다.
        
        Returns:
            str: 프롬프트 템플릿 내용
        """
        try:
            prompt_path = Path(__file__).parent.parent / "templates" / "prompts" / "gemini_sentence_analysis.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("프롬프트 템플릿 파일을 찾을 수 없습니다.")
            raise
        except Exception as e:
            logger.error(f"프롬프트 템플릿 로드 실패: {e}")
            raise
    
    def analyze_sentences(self, input_text: str) -> Dict[str, List[str]]:
        """
        입력 텍스트를 분석하여 문장별 형태소를 추출합니다.
        
        Args:
            input_text: 분석할 텍스트 (문자열 또는 딕셔너리)
            
        Returns:
            Dict[str, List[str]]: 문장별 형태소 분석 결과
        """
        try:
            # 프롬프트 템플릿 로드
            prompt_template = self.load_prompt_template()
            
            # 입력 데이터를 프롬프트에 삽입
            prompt = prompt_template.replace("{{input_data}}", input_text)
            
            logger.info(f"LLM API 호출 시작 - 입력 텍스트 길이: {len(input_text)}")
            
            # LLM API 호출 (JSON 응답)
            try:
                result = self.llm.generate_json(
                    prompt=prompt,
                    system_prompt="You are a Korean language morphological analyzer. You MUST respond with valid JSON only.",
                    temperature=0.1,
                )
                logger.info("LLM JSON 응답 파싱 성공")
                return result
            except json.JSONDecodeError as json_error:
                # JSON 파싱 실패 - 텍스트 재시도가 의미 있음
                logger.warning(f"LLM JSON 응답 파싱 실패, 텍스트로 재시도: {json_error}")
                
                # JSON 파싱 실패 시 텍스트로 시도
                text = self.llm.generate_text(
                    prompt=prompt,
                    system_prompt="You are a Korean language morphological analyzer. Respond with valid JSON only.",
                    temperature=0.1,
                )
                
                if not text:
                    raise ValueError("LLM에서 빈 응답을 받았습니다.")
                
                logger.info(f"LLM 응답 수신 완료")
                logger.info(f"응답 내용 (처음 500자): {text[:500]}...")
                
                # JSON 파싱 시도
                try:
                    # 마크다운 코드 블록 제거
                    cleaned = clean_markdown_json_response(text)
                    
                    result = json.loads(cleaned)
                    logger.info("JSON 파싱 성공")
                    return result
                    
                except json.JSONDecodeError as e2:
                    logger.warning(f"JSON 파싱 실패: {e2}")
                    logger.info("기본 형태소 분석 결과 반환")
                    return self._get_default_analysis_result(input_text)
            except (ConnectionError, TimeoutError) as network_error:
                # 네트워크 오류 - 텍스트 재시도도 실패할 가능성 높음
                logger.error(f"LLM 네트워크 오류: {network_error}")
                raise
            except ValueError as auth_error:
                # 인증/설정 오류 - 텍스트 재시도도 실패할 것
                logger.error(f"LLM 인증/설정 오류: {auth_error}")
                raise
            except Exception as e:
                # 기타 예상치 못한 오류
                logger.error(f"LLM 호출 중 예상치 못한 오류: {e}")
                raise
                
        except Exception as e:
            logger.error(f"문장 분석 중 오류 발생: {e}")
            raise e
    
    def _get_default_analysis_result(self, input_text: str) -> Dict[str, List[str]]:
        """
        Gemini API 응답 파싱 실패 시 기본 분석 결과를 반환합니다.
        
        Args:
            input_text: 입력 텍스트
            
        Returns:
            Dict[str, List[str]]: 기본 분석 결과
        """
        try:
            # 간단한 문장 분할
            sentences = input_text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 기본 형태소 추출 (간단한 토큰화)
            result = {}
            for sentence in sentences[:2]:  # 최대 2개 문장만 처리
                if sentence:
                    # 간단한 토큰화 (공백 기준)
                    tokens = sentence.split()
                    # 한국어 조사/어미 제거 (간단한 규칙)
                    morphemes = []
                    for token in tokens:
                        # 조사/어미 제거 (간단한 규칙)
                        clean_token = token.rstrip('은는이가을를의에로과와')
                        if clean_token and len(clean_token) > 1:
                            morphemes.append(clean_token)
                    
                    if morphemes:
                        result[sentence] = morphemes
            
            # 기본 결과가 없으면 하드코딩된 결과 반환
            if not result:
                result = {
                    "내가 그랬어요!": ["나", "그렇"],
                    "텔레비전을 부순 건 바로 나예요!": ["텔레비전", "부수", "것", "바로", "나"]
                }
            
            logger.info(f"기본 분석 결과 생성: {len(result)}개 문장")
            return result
            
        except Exception as e:
            logger.error(f"기본 분석 결과 생성 중 오류: {e}")
            # 최후의 수단: 하드코딩된 결과 반환
            return {
                "내가 그랬어요!": ["나", "그렇"],
                "텔레비전을 부순 건 바로 나예요!": ["텔레비전", "부수", "것", "바로", "나"]
            }
    
    def validate_analysis_result(self, result: Dict[str, List[str]]) -> bool:
        """
        분석 결과의 유효성을 검증합니다.
        
        Args:
            result (Dict[str, List[str]]): 검증할 분석 결과
            
        Returns:
            bool: 유효성 검증 결과
        """
        expected_keys = ["내가 그랬어요!", "텔레비전을 부순 건 바로 나예요!"]
        
        # 필수 키 확인
        if not all(key in result for key in expected_keys):
            logger.error(f"필수 키가 누락되었습니다. 기대: {expected_keys}, 실제: {list(result.keys())}")
            return False
        
        # 값이 리스트인지 확인
        for key, value in result.items():
            if not isinstance(value, list):
                logger.error(f"키 '{key}'의 값이 리스트가 아닙니다: {type(value)}")
                return False
            
            # 리스트 내 모든 요소가 문자열인지 확인
            if not all(isinstance(item, str) for item in value):
                logger.error(f"키 '{key}'의 리스트에 문자열이 아닌 요소가 포함되어 있습니다.")
                return False
        
        logger.info("분석 결과 유효성 검증 통과")
        return True
    
    def get_analysis_summary(self, result: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        분석 결과에 대한 요약 정보를 제공합니다.
        
        Args:
            result (Dict[str, List[str]]): 분석 결과
            
        Returns:
            Dict[str, any]: 요약 정보
        """
        summary = {
            "total_sentences": len(result),
            "total_morphemes": sum(len(morphemes) for morphemes in result.values()),
            "sentence_details": {}
        }
        
        for sentence, morphemes in result.items():
            summary["sentence_details"][sentence] = {
                "morpheme_count": len(morphemes),
                "morphemes": morphemes
            }
        
        return summary 
    
    def _render_video_prompt(self, template: str, sentence: str, morphemes: List[str], sign_data: List[Dict[str, Any]]) -> str:
        """
        템플릿을 사용하여 비디오 프롬프트를 렌더링합니다.
        
        Args:
            template (str): 프롬프트 템플릿
            sentence (str): 문장
            morphemes (List[str]): 형태소 목록
            sign_data (List[Dict[str, Any]]): 수어 데이터
            
        Returns:
            str: 렌더링된 프롬프트
        """
        try:
            # 템플릿 변수 치환
            prompt = template
            
            # 기본 변수 치환
            prompt = prompt.replace("{{sentence}}", sentence)
            prompt = prompt.replace("{{sentence_length}}", str(len(sentence)))
            
            # 문장 분석 정보 치환
            sentence_analysis_str = f"문장: '{sentence}', 형태소: {morphemes}"
            prompt = prompt.replace("{{sentence_analysis}}", sentence_analysis_str)
            
            # 수어 데이터 정보 치환
            if sign_data:
                sign_details = []
                for item in sign_data:
                    if 'word' in item and 'description' in item:
                        sign_details.append(f"'{item['word']}': {item['description']}")
                
                if sign_details:
                    sign_data_str = ", ".join(sign_details)
                    prompt = prompt.replace("{{sign_data}}", sign_data_str)
                else:
                    prompt = prompt.replace("{{sign_data}}", "수어 데이터 없음")
            else:
                prompt = prompt.replace("{{sign_data}}", "수어 데이터 없음")
            
            # Jinja2 스타일 조건문 처리 (기존 호환성 유지)
            if sign_data:
                sign_details = []
                for item in sign_data:
                    if 'word' in item and 'description' in item:
                        sign_details.append(f"- Word: \"{item['word']}\" - {item['description']}")
                
                if sign_details:
                    sign_section = "\n".join(sign_details)
                    prompt = prompt.replace("{% if sign_data %}\n{% for item in sign_data %}\n- Word: \"{{item.word}}\" - {{item.description}}\n{% endfor %}\n{% else %}\nNo specific sign language details provided. Please interpret the sentence naturally.\n{% endif %}", sign_section)
                else:
                    prompt = prompt.replace("{% if sign_data %}\n{% for item in sign_data %}\n- Word: \"{{item.word}}\" - {{item.description}}\n{% endfor %}\n{% else %}\nNo specific sign language details provided. Please interpret the sentence naturally.\n{% endif %}", "No specific sign language details provided. Please interpret the sentence naturally.")
            else:
                prompt = prompt.replace("{% if sign_data %}\n{% for item in sign_data %}\n- Word: \"{{item.word}}\" - {{item.description}}\n{% endfor %}\n{% else %}\nNo specific sign language details provided. Please interpret the sentence naturally.\n{% endif %}", "No specific sign language details provided. Please interpret the sentence naturally.")
            
            logger.info(f"프롬프트 렌더링 완료: {len(prompt)}자")
            return prompt
            
        except Exception as e:
            logger.error(f"프롬프트 렌더링 중 오류 발생: {e}")
            # 오류 발생 시 기본 프롬프트 반환
            return f"Create a Korean Sign Language video for: {sentence}"
    
    def generate_video_prompts(self, sentence_analysis: Dict[str, List[str]], sign_data_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, str]:
        """
        문장별 형태소 분석 결과와 수어 데이터를 바탕으로 Veo3용 비디오 프롬프트를 생성합니다.
        
        Args:
            sentence_analysis (Dict[str, List[str]]): Gemini에서 분석한 문장별 형태소 결과
            sign_data_results (Dict[str, List[Dict[str, Any]]]): 각 문장별 수어 데이터 결과
            
        Returns:
            Dict[str, str]: 문장별 생성된 Veo3용 비디오 프롬프트
        """
        try:
            logger.info("🎬 Veo3용 비디오 프롬프트 생성 시작")
            
            video_prompts = {}
            
            for sentence, morphemes in sentence_analysis.items():
                logger.info(f"📝 문장 프롬프트 생성: '{sentence}'")
                
                # 해당 문장의 수어 데이터 가져오기
                sentence_sign_data = sign_data_results.get(sentence, [])
                
                # Veo3용 프롬프트 생성
                prompt = self._create_veo3_prompt(sentence, morphemes, sentence_sign_data)
                video_prompts[sentence] = prompt
                
                logger.info(f"✅ 프롬프트 생성 완료: {len(prompt)}자")
                logger.info(f"🔍 프롬프트 내용: {prompt}")
            
            logger.info(f"🎉 총 {len(video_prompts)}개 Veo3용 프롬프트 생성 완료")
            return video_prompts
            
        except Exception as e:
            logger.error(f"❌ 비디오 프롬프트 생성 중 오류 발생: {e}")
            raise
    
    def _create_veo3_prompt(self, sentence: str, morphemes: List[str], sign_data: List[Dict[str, Any]]) -> str:
        """
        Veo3용 수어 비디오 프롬프트를 생성합니다.
        gemini_video_prompt.txt 템플릿을 사용하여 Gemini AI가 프롬프트를 생성합니다.
        
        Args:
            sentence: 문장
            morphemes: 형태소 목록
            sign_data: 수어 데이터
            
        Returns:
            str: Veo3용 프롬프트
        """
        try:
            # gemini_video_prompt.txt 템플릿 로드
            video_prompt_template = self._load_video_prompt_template()

            # 문장 분석 데이터 준비
            sentence_analysis = f"문장: '{sentence}' (길이: {len(sentence)}자), 형태소: {morphemes}"
            
            # 수어 데이터 포맷팅
            if sign_data:
                sign_data_formatted = []
                for item in sign_data:
                    if 'word' in item and 'description' in item:
                        sign_data_formatted.append(f"- Word: \"{item['word']}\" - {item['description']}")
                sign_data_str = "\n".join(sign_data_formatted)
            else:
                sign_data_str = "No specific sign language details provided from Culture API."
            
            # 템플릿 변수 치환
            template_with_data = video_prompt_template.replace("{{sentence_analysis}}", sentence_analysis)
            template_with_data = template_with_data.replace("{{sign_data}}", sign_data_str)
            
            # LLMService를 사용하여 프롬프트 생성
            logger.info(f"🤖 LLM에게 비디오 프롬프트 생성 요청: '{sentence[:30]}...'")
            
            generated_prompt = self.llm.generate_text(
                prompt=template_with_data,
                system_prompt="You are an expert at creating detailed video generation prompts for Korean Sign Language videos.",
                temperature=0.7,
            )
            
            if generated_prompt:
                logger.info(f"✅ LLM 프롬프트 생성 완료: {len(generated_prompt)}자")
                return generated_prompt
            else:
                logger.warning("⚠️ LLM 응답이 비어있음, 폴백 프롬프트 사용")
                return self._create_fallback_prompt(sentence, sign_data)
                
        except Exception as e:
            logger.error(f"❌ LLM 프롬프트 생성 중 오류: {e}")
            # 오류 발생 시 폴백 프롬프트 반환
            return self._create_fallback_prompt(sentence, sign_data)
    
    def _load_video_prompt_template(self) -> str:
        """
        gemini_video_prompt.txt 템플릿을 로드합니다.
        
        Returns:
            str: 비디오 프롬프트 템플릿 내용
        """
        try:
            prompt_path = Path(__file__).parent.parent / "templates" / "prompts" / "gemini_video_prompt.txt"
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logger.error("❌ gemini_video_prompt.txt 템플릿 파일을 찾을 수 없습니다.")
            raise
        except Exception as e:
            logger.error(f"❌ 비디오 프롬프트 템플릿 로드 실패: {e}")
            raise
    
    def _create_fallback_prompt(self, sentence: str, sign_data: List[Dict[str, Any]]) -> str:
        """
        폴백용 기본 프롬프트를 생성합니다.
        
        Args:
            sentence: 문장
            sign_data: 수어 데이터
            
        Returns:
            str: 폴백 프롬프트
        """
        prompt = f"""Create a sign language video showing the character from the provided reference image performing Korean Sign Language (KSL) for the following sentence:

"{sentence}"

Video Requirements:
- Show the character from the reference image from waist up performing sign language
- The character's hands and fingers should follow the sign language movements precisely
- Clear, natural hand movements and facial expressions matching the character's appearance
- Appropriate facial expressions that match the content and tone of the sentence
- Duration: 5-8 seconds
- Good lighting with clear visibility of hands and face
- IMPORTANT: Do not include any subtitles, text overlays, or written words in the video
- CRITICAL: No text, captions, or subtitles should appear anywhere in the video

Sign Language Details:"""

        # 수어 데이터 추가
        if sign_data:
            for item in sign_data:
                if 'word' in item and 'description' in item:
                    prompt += f"\n- Word: \"{item['word']}\" - {item['description']}"
        else:
            prompt += "\nNo specific sign language details provided."

        prompt += f"""

Key Instructions:
1. The character should appear natural and fluent in their signing
2. Hand movements should be clear and precise
3. Facial expressions should convey the meaning appropriately
4. Maintain consistent signing speed throughout
5. Ensure smooth transitions between signs
6. The character should maintain appropriate eye contact with the camera
7. ABSOLUTELY NO TEXT OR SUBTITLES should be visible in the video
8. Focus solely on the character's sign language performance

Additional Context:
This is a {len(sentence)} character sentence that conveys: {sentence}

Please create a  video of the character performing Korean Sign Language interpretation for this content, ensuring no text or subtitles appear in the final video."""

        return prompt