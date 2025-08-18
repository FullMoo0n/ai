#!/bin/bash

echo "Redis 서버를 시작합니다..."

# Docker Compose로 Redis 시작
docker-compose up -d redis

echo "Redis 서버가 시작되었습니다."
echo "Redis Commander는 http://localhost:8081 에서 확인할 수 있습니다."
echo ""
echo "Redis 연결 확인:"
echo "  Host: localhost"
echo "  Port: 6379"
echo "  Database: 0" 