#!/bin/bash

echo "Celery 워커를 시작합니다..."

# 현재 디렉토리를 프로젝트 루트로 설정
cd "$(dirname "$0")/.."

# 환경 변수 로드
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Celery 워커 시작
celery -A app.celery_app worker --loglevel=info --concurrency=4

echo "Celery 워커가 종료되었습니다." 