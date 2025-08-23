FROM python:3.10-slim

# 1. 작업 디렉토리 설정
WORKDIR /app

# 2. requirements.txt만 먼저 복사 (캐시 활용)
COPY requirements.txt .

# 3. 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 4. 실제 코드 복사 (변경 가능성이 높기 때문에 나중에 복사해야 pip 캐시 유지됨)
COPY app ./app

# 4-1. 이미지 파일 복사
COPY images ./images

# 5. PYTHONPATH 환경변수 설정 (모듈 import를 위해)
ENV PYTHONPATH=/app

# 6. Celery 워커를 백그라운드에서 실행한 후 FastAPI 서버 실행
CMD bash -c "celery -A app.celery_app worker --loglevel=info & uvicorn app.main:app --host 0.0.0.0 --port 8000"
