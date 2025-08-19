#!/bin/bash

echo "=== AI 비디오 생성 서비스 시작 ==="
echo ""

# 현재 디렉토리를 프로젝트 루트로 설정
cd "$(dirname "$0")/.."

# Redis 시작
# echo "1. Redis 서버 시작..."
# docker-compose up -d redis
# sleep 3

# 환경 변수 확인
echo "2. 환경 변수 확인..."
if [ ! -f .env ]; then
    echo "⚠️  .env 파일이 없습니다. .env.example을 참고하여 .env 파일을 생성하세요."
    exit 1
fi

# Python 의존성 확인
echo "3. Python 의존성 확인..."
python -c "import celery, redis" 2>/dev/null || {
    echo "⚠️  필요한 Python 패키지가 설치되지 않았습니다."
    echo "다음 명령어로 설치하세요: pip install -r requirements.txt"
    exit 1
}

echo ""
echo "=== 서비스 시작 완료 ==="
echo ""
echo "다음 명령어로 각 컴포넌트를 실행하세요:"
echo ""
echo "1. FastAPI 서버:"
echo "   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "2. Celery 워커 (새 터미널):"
echo "   ./scripts/start_celery.sh"
echo ""
echo "3. Flower 모니터링 (선택사항, 새 터미널):"
echo "   ./scripts/start_flower.sh"
echo ""
echo "=== 접속 정보 ==="
echo "FastAPI 문서: http://localhost:8000/docs"
echo "Redis Commander: http://localhost:8081"
echo "Flower 모니터링: http://localhost:5555 (Flower 시작 후)" 