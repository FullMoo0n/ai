import os
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class SignDataService:
    def __init__(self):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = "sign_languages"
        
        # Qdrant Client 초기화
        try:
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self.client = None

        # OpenAI Client 초기화 (임베딩용)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found. Semantic search will not work.")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=self.openai_api_key)

    def _get_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환"""
        if not self.openai_client:
            raise ValueError("OpenAI Client not initialized")
        
        text = text.replace("\n", " ")
        response = self.openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def search_sign_description(self, keyword: str, limit: int = 1) -> Optional[str]:
        """
        키워드로 수어 설명을 검색합니다.
        
        Args:
            keyword: 검색할 단어
            limit: 반환할 결과 수
            
        Returns:
            가장 적합한 수어 설명 (없으면 None)
        """
        if not self.client:
            logger.error("Qdrant client is not available")
            return None
            
        try:
            # 1. 키워드 임베딩
            query_vector = self._get_embedding(keyword)
            
            # 2. 벡터 검색
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=0.7  # 유사도 임계값 설정
            )
            search_result = response.points
            
            if not search_result:
                logger.info(f"No sign description found for '{keyword}'")
                return None
                
            # 가장 높은 점수의 결과 반환
            top_result = search_result[0]
            description = top_result.payload.get("description")
            
            logger.info(f"Found match for '{keyword}': {top_result.payload.get('keywords')} (Score: {top_result.score})")
            return description

        except Exception as e:
            logger.error(f"Error searching sign description: {e}")
            return None
            
    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 확인"""
        status = {
            "qdrant_connection": False,
            "openai_connection": False,
            "collection_exists": False,
            "document_count": 0
        }
        
        if self.client:
            try:
                collections = self.client.get_collections()
                status["qdrant_connection"] = True
                
                # 컬렉션 존재 여부 확인
                for col in collections.collections:
                    if col.name == self.collection_name:
                        status["collection_exists"] = True
                        # 문서 수 확인
                        count_result = self.client.count(collection_name=self.collection_name)
                        status["document_count"] = count_result.count
                        break
            except Exception:
                pass
                
        if self.openai_client:
            status["openai_connection"] = True
            
        return status
