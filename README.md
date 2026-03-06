# AI 비디오 생성 API

OpenAI(Sora/LLM/Vision)를 활용한 수어 비디오 생성 API와 OCR, 문장 분리 기능을 제공합니다.

## 주요 기능

- **OCR**: 이미지에서 텍스트 추출 및 문단 분석
- **문장 분리**: 텍스트를 문장 단위로 분리
- **토큰화**: 문장을 단어 단위로 분리
- **문장 검증**: OpenAI를 활용한 문장 분리 품질 검증
- **문장 분석**: LLM(Ollama/OpenAI)을 활용한 한국어 문장 형태소 분석 및 비디오 프롬프트 생성
- **비디오 생성**: OpenAI Sora 기반 텍스트-to-비디오 생성
- **Azure 업로드**: 생성된 비디오를 Azure Blob Storage에 업로드

## 현재 파이프라인/프롬프트 맵 (2026-03-05 기준)

아래는 **현재 코드 기준으로 실제 연결된 흐름**입니다.

### 1) 이미지 → 수어 비디오 통합 파이프라인 (권장)

- 엔드포인트: `POST /process-image-to-videos`
- 진입 코드: `app/main.py` → `IntegratedPipeline.execute()`
- 핵심 단계:
  1. OCR (이미지 텍스트 + image context + character_description 추출)
  2. 문장 분할
  3. 전체 문장 통합 토큰 처리 + Qdrant RAG 수어데이터 수집
  4. gloss_sequence 생성 + 통합 프롬프트 생성
  5. Sora 비디오 생성

사용 프롬프트:
- `app/templates/prompts/gemini_video_prompt.txt`
  - 사용 위치: `app/services/prompt_template.py`의 `build_video_prompt()`
  - 설명: `original_text`, `gloss_sequence`, `sign_data`, `image_context`, `character_description`, 메타데이터를 템플릿에 직접 치환해 데모용 프롬프트를 생성

### 2) 이미지 → 수어 비디오 비동기 파이프라인

- 엔드포인트: `POST /process-image-to-videos-legacy`
- 진입 코드: `app/main.py` → `SyncIntegratedPipeline.execute()`
- 핵심 단계:
  1. OCR
  2. 전체 텍스트 토큰/수어데이터 수집
  3. 통합 프롬프트 생성
  4. Sora 비디오 생성

주의:
- `SyncIntegratedPipeline.execute()` 내부에서 `_step_process_text` 호출이 보이며, 현재 구현 본문은 `_step_tokenize_and_process` / `_step_generate_prompt`로 분리되어 있습니다.
- 레거시 파이프라인을 운영 경로로 사용할 경우, 코드 동기화 상태를 먼저 확인하세요.

사용 프롬프트:
- `app/templates/prompts/gemini_video_prompt.txt`
  - 사용 위치: `app/services/prompt_template.py`의 `build_video_prompt()`

### 2-1) 기능별 테스트 엔드포인트

- `POST /ocr`: OCR 단독 테스트
- `POST /sentences`: 문장 분리 단독 테스트
- `POST /tokens`: 토큰화 단독 테스트
- `POST /culture/sign-description`: Qdrant RAG(임베딩 검색) 단독 테스트
- `POST /sentences/validate`: OpenAI 기반 분리 검증
- `POST /process-image-to-videos`: 전체 통합 파이프라인 테스트 (권장)
- `POST /process-image-to-videos-legacy`: 레거시 파이프라인 테스트
- `GET /pipeline-status/{task_id}`: 파이프라인 상태 확인
- `POST /analyze-sentences`: 형태소 분석 전용 테스트

### 3) 문장 분석 파이프라인

- 엔드포인트: `POST /analyze-sentences`
- 진입 코드: `app/main.py` → `GeminiService.analyze_sentences()`

사용 프롬프트:
- `app/templates/prompts/gemini_sentence_analysis.txt`
  - 사용 위치: `app/services/gemini_service.py`의 `load_prompt_template()`
  - 설명: 입력 텍스트를 규칙 기반 JSON 형태소 분석 결과로 유도

### 현재 사용 중인 프롬프트 파일 (정리 완료)

- `app/templates/prompts/gemini_video_prompt.txt`
- `app/templates/prompts/gemini_sentence_analysis.txt`

삭제된 미사용 템플릿:
- `command_prompt.txt`
- `question_prompt.txt`
- `statement_prompt.txt`
- `default_prompt.txt`

## 아키텍처 메모 (Sora/OpenAI 중심)

- 비디오 생성은 `app/services/sora_service.py`를 통해 **OpenAI Sora**로 수행됩니다.
- OCR은 OpenAI Vision 기반 경로를 사용하며, 생성된 텍스트/이미지 맥락/캐릭터 설명이 프롬프트 생성에 반영됩니다.
- 최종 비디오는 Azure Blob Storage에 업로드되어 URL로 반환됩니다.
- 문장 분석은 `GeminiService`라는 이름을 유지하지만 내부적으로 `LLMService`(OpenAI/Ollama)를 사용합니다.
- 데모용 비디오 프롬프트는 LLM 재작성 없이 `build_video_prompt()`에서 템플릿 직접 치환 방식으로 생성합니다.

## 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env.example`을 참고하여 `.env` 파일을 생성하세요:

```bash
cp .env.example .env
# .env 파일을 편집하여 API 키들을 설정
```

필수 환경 변수:
- `LLM_PROVIDER`: LLM 제공자 (`ollama` 또는 `openai`, 기본: `openai`)
- `OPENAI_MODEL`: OpenAI 모델명 (기본: `gpt-4o`)
- `OLLAMA_BASE_URL`: Ollama API 주소 (기본: `http://localhost:11434/v1`)
- `OLLAMA_MODEL`: Ollama 모델명 (기본: `qwen2.5:7b-instruct-q4_K_M`)
- `OPENAI_API_KEY`: OpenAI API 키 (Sora/LLM/Vision 동작용)
- `REDIS_URL`: Redis 연결 URL (기본: redis://localhost:6379/0)
- `AUTO_INGEST_SIGN_DATA_ON_QDRANT_ACCESS`: Qdrant 조회 시 데이터가 비어 있으면 CSV 자동 적재 (`true`/`false`, 기본: `true`)

Azure 업로드를 위한 환경 변수:
- `AZURE_STORAGE_CONNECTION_STRING`: Azure Blob Storage 연결 문자열
- `AZURE_CONTAINER_NAME`: 업로드 대상 컨테이너명 (기본: `blob-binary`)

기타 외부 연동 환경 변수:
- `CULTURE_API_KEY`: 수어 설명 조회용 한국문화정보원 API 키
- `VISION_API_KEY`: Google Vision OCR 경로 사용 시 필요

### 3. Redis 서버 시작

```bash
# Docker Compose로 Redis 시작
# docker-compose up -d redis

# 또는 스크립트 사용
# ./scripts/start_redis.sh
```

### 4. 서비스 시작

#### 옵션 1: 통합 스크립트 사용

```bash
./scripts/start_all.sh
```

이후 각 컴포넌트를 개별 터미널에서 실행:

```bash
# 터미널 1: FastAPI 서버
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 터미널 2: Celery 워커
./scripts/start_celery.sh

# 터미널 3 (선택사항): Flower 모니터링
./scripts/start_flower.sh
```

#### 옵션 2: 개별 실행

```bash
# 1. Redis 시작
docker-compose up -d redis

# 2. FastAPI 서버 (터미널 1)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 3. Celery 워커 (터미널 2)
celery -A app.celery_app worker --loglevel=info --concurrency=4

# 4. Flower 모니터링 (터미널 3, 선택사항)
celery -A app.celery_app flower --port=5555
```

## API 사용법

### 1) 이미지 → 수어 비디오 생성 (권장)

```bash
curl -X POST "http://localhost:8000/process-image-to-videos" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_image_url": "https://your-bucket.s3.amazonaws.com/your-image.jpg"
  }'
```

### 2) 레거시 파이프라인 호출

```bash
curl -X POST "http://localhost:8000/process-image-to-videos-legacy" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_image_url": "https://your-bucket.s3.amazonaws.com/your-image.jpg"
  }'
```

### 3) 파이프라인 상태 조회

```bash
curl "http://localhost:8000/pipeline-status/{task_id}"
```

### 4) 문장 분석 (LLM 통합)

설정된 LLM(Ollama 또는 OpenAI)을 사용하여 한국어 문장을 분석하고 핵심 형태소를 추출합니다:

```bash
curl -X POST "http://localhost:8000/analyze-sentences" \
  -H "Content-Type: application/json" \
  -d '"내가 그랬어요! 텔레비전을 부순 건 바로 나예요!"'
```

응답:
```json
{
  "success": true,
  "input_text": "내가 그랬어요! 텔레비전을 부순 건 바로 나예요!",
  "analysis_result": {
    "내가 그랬어요!": ["나", "그렇"],
    "텔레비전을 부순 건 바로 나예요!": ["텔레비전", "부수", "것", "바로", "나"]
  },
  "summary": {
    "total_sentences": 2,
    "total_morphemes": 7,
    "sentence_details": {
      "내가 그랬어요!": {
        "morpheme_count": 2,
        "morphemes": ["나", "그렇"]
      },
      "텔레비전을 부순 건 바로 나예요!": {
        "morpheme_count": 5,
        "morphemes": ["텔레비전", "부수", "것", "바로", "나"]
      }
    }
  }
}
```

**분석 규칙:**
- 문장을 두 개로 분할: "내가 그랬어요!"와 "텔레비전을 부순 건 바로 나예요!"
- 각 문장에서 핵심 형태소만 추출 (명사, 대명사, 동사/형용사 어간, 부사)
- 조사(-가, -을)와 어미(-어요) 등 문법 요소는 제외
- 동사 어간의 끝 하이픈(-) 제거

## 모니터링

### 접속 정보

- **FastAPI 문서**: http://localhost:8000/docs
- **Redis Commander**: http://localhost:8081 (Redis 관리)
- **Flower**: http://localhost:5555 (Celery 작업 모니터링)

### Flower에서 확인 가능한 정보

- 진행 중인 작업 목록
- 완료된 작업 통계
- 워커 상태 및 성능
- 실시간 작업 진행 상황

## 작업 상태

| 상태 | 설명 |
|------|------|
| `PENDING` | 작업이 큐에서 대기 중 |
| `PROGRESS` | 작업이 진행 중 (세부 상황 정보 포함) |
| `SUCCESS` | 작업 완료 (비디오 URI 포함) |
| `FAILURE` | 작업 실패 (오류 메시지 포함) |
| `RETRY` | 작업 재시도 중 |
| `REVOKED` | 작업이 취소됨 |

## 트러블슈팅

### Redis 연결 오류

```bash
# Redis 상태 확인
docker ps | grep redis

# Redis 재시작
docker-compose restart redis
```

### Celery 워커 오류

```bash
# 워커 로그 확인
celery -A app.celery_app worker --loglevel=debug

# 워커 재시작
pkill -f celery
./scripts/start_celery.sh
```

### 환경 변수 확인

```bash
# .env 파일 내용 확인
cat .env

# Python에서 환경 변수 확인
python -c "import os; print('OPENAI_API_KEY:', bool(os.getenv('OPENAI_API_KEY')))"
```

## 성능 최적화

### Celery 워커 조정

```bash
# 동시 작업 수 조정 (기본: 4)
celery -A app.celery_app worker --concurrency=8

# 특정 큐만 처리
celery -A app.celery_app worker --queues=video_generation
```

### Redis 메모리 관리

```bash
# Redis 메모리 사용량 확인
docker exec ai_redis redis-cli info memory

# 만료된 키 정리
docker exec ai_redis redis-cli flushdb
```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 