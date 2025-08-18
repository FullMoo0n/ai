# AI 비디오 생성 API

Google Veo 3를 활용한 텍스트-to-비디오 생성 API와 OCR, 문장 분리 기능을 제공합니다.

## 주요 기능

- **OCR**: 이미지에서 텍스트 추출 및 문단 분석
- **문장 분리**: 텍스트를 문장 단위로 분리
- **토큰화**: 문장을 단어 단위로 분리
- **문장 검증**: OpenAI를 활용한 문장 분리 품질 검증
- **비디오 생성**: Google Veo 3를 활용한 텍스트-to-비디오 생성 (동기/비동기)
- **S3 업로드**: 생성된 비디오를 자동으로 AWS S3에 업로드

## 새로운 기능: 비동기 비디오 생성

### 기존 문제점
- Veo 3 비디오 생성 시 긴 대기 시간으로 인한 타임아웃 발생
- 동기 방식으로 인한 서버 리소스 점유

### 해결책
- **Celery + Redis**를 활용한 비동기 작업 처리
- 즉시 task_id 반환 후 별도 상태 조회 방식
- 실시간 진행 상황 모니터링

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
- `GOOGLE_API_KEY`: Google Gemini API 키 (Veo 비디오 생성용)
- `OPENAI_API_KEY`: OpenAI API 키 (문장 검증용)
- `REDIS_URL`: Redis 연결 URL (기본: redis://localhost:6379/0)

S3 업로드를 위한 환경 변수:
- `AWS_ACCESS_KEY_ID`: AWS 액세스 키 ID
- `AWS_SECRET_ACCESS_KEY`: AWS 시크릿 액세스 키
- `AWS_REGION`: AWS 리전 (기본: ap-northeast-2)
- `S3_BUCKET_NAME`: S3 버킷 이름

### 3. Redis 서버 시작

```bash
# Docker Compose로 Redis 시작
docker-compose up -d redis

# 또는 스크립트 사용
./scripts/start_redis.sh
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

### 비동기 비디오 생성 (권장)

#### 1. 비디오 생성 요청

```bash
curl -X POST "http://localhost:8000/veo/async" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A majestic eagle soaring over snow-capped mountains at sunset",
    "aspect_ratio": "16:9",
    "timeout_seconds": 600
  }'
```

응답:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "queued",
  "message": "비디오 생성 작업이 큐에 추가되었습니다. `/veo/status/{task_id}`로 상태를 확인하세요."
}
```

#### 2. 작업 상태 조회

```bash
curl "http://localhost:8000/veo/status/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

진행 중인 경우:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "PROGRESS",
  "result": null,
  "progress": {
    "status": "processing",
    "message": "비디오 생성 진행중... (120초 경과)",
    "operation": "operations/generate-video-12345",
    "elapsed_seconds": 120
  },
  "error": null
}
```

완료된 경우:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "SUCCESS",
  "result": {
    "status": "completed",
    "video_uri": "https://storage.googleapis.com/your-video-uri",
    "operation": "operations/generate-video-12345",
    "message": "비디오 생성 완료. Gemini API를 통해 직접 다운로드 가능합니다.",
    "error": null
  },
  "progress": null,
  "error": null
}
```

#### 3. 작업 취소 (선택사항)

```bash
curl -X DELETE "http://localhost:8000/veo/cancel/a1b2c3d4-e5f6-7890-abcd-ef1234567890"
```

### 동기 비디오 생성

기존 방식이 유지됩니다:

```bash
curl -X POST "http://localhost:8000/veo" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A red panda riding a skateboard in a sunny park",
    "aspect_ratio": "16:9",
    "timeout_seconds": 180
  }'
```

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
python -c "import os; print('GOOGLE_API_KEY:', bool(os.getenv('GOOGLE_API_KEY')))"
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