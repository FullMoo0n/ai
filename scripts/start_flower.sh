#!/bin/bash

echo "Flower (Celery 모니터링)를 시작합니다..."

# 현재 디렉토리를 프로젝트 루트로 설정
cd "$(dirname "$0")/.."

# 환경 변수 로드
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Flower 시작
celery -A app.celery_app flower --port=5555

echo "Flower가 종료되었습니다." 