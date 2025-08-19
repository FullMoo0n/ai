import os
from celery import Celery
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Redis URL 설정
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery 인스턴스 생성
celery_app = Celery(
    "ai_video_generator",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["app.tasks.veo_tasks"]  # 태스크 모듈 포함
)

# Celery 설정
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    result_expires=3600,  # 결과를 1시간 동안 저장
    task_time_limit=900,  # 태스크 타임아웃 15분
    task_soft_time_limit=600,  # 소프트 타임아웃 10분
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)
