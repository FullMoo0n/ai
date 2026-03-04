import os
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import logging
import threading
from dotenv import load_dotenv

# 로거 초기화
logger = logging.getLogger(__name__)

load_dotenv()

# 기본 유사도 임계값
DEFAULT_SCORE_THRESHOLD = 0.7


class SignDataService:
    def __init__(
        self,
        score_threshold: Optional[float] = None,
    ):
        self.qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        self.qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = "sign_languages"
        self.score_threshold = score_threshold or float(
            os.getenv("QDRANT_SCORE_THRESHOLD", DEFAULT_SCORE_THRESHOLD)
        )
        self.auto_ingest_on_access = os.getenv(
            "AUTO_INGEST_SIGN_DATA_ON_QDRANT_ACCESS", "true"
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        self._ingest_checked = False
        self._ingest_lock = threading.Lock()

        # Qdrant Client 초기화 (fail-fast)
        try:
            self.client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
            logger.info(f"Connected to Qdrant at {self.qdrant_host}:{self.qdrant_port}")
        except Exception as e:
            raise RuntimeError(
                f"Qdrant 클라이언트 초기화 실패 ({self.qdrant_host}:{self.qdrant_port}): {e}"
            ) from e

        # OpenAI Client 초기화 (fail-fast)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. 임베딩 검색을 사용할 수 없습니다."
            )
        self.openai_client = OpenAI(api_key=self.openai_api_key)

    def _get_collection_count(self) -> int:
        """sign_languages 컬렉션 포인트 개수 조회 (없거나 오류 시 0)"""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            if not exists:
                return 0
            return int(
                self.client.count(
                    collection_name=self.collection_name,
                    exact=True,
                ).count
            )
        except Exception as e:
            logger.warning(f"Qdrant 컬렉션 카운트 조회 실패: {e}")
            return 0

    def _ensure_collection_populated_if_needed(self) -> None:
        """Qdrant 접근 시점에 컬렉션이 비어 있으면 CSV 자동 적재"""
        if self._ingest_checked:
            return

        if not self.auto_ingest_on_access:
            self._ingest_checked = True
            return

        with self._ingest_lock:
            if self._ingest_checked:
                return

            existing_count = self._get_collection_count()
            if existing_count > 0:
                logger.info(
                    f"Qdrant 기존 데이터 감지: {existing_count}건 (자동 적재 생략)"
                )
                self._ingest_checked = True
                return

            logger.info("Qdrant 데이터 없음 → 접근 시점 자동 적재 시작")
            try:
                from scripts.ingest_sign_data import ingest_data

                ingest_data(recreate_collection=False)
                ingested_count = self._get_collection_count()
                logger.info(f"Qdrant 접근 시점 자동 적재 완료: {ingested_count}건")
            except Exception as e:
                logger.error(f"Qdrant 접근 시점 자동 적재 실패: {e}")
            finally:
                self._ingest_checked = True

    def _get_embedding(self, text: str) -> List[float]:
        """텍스트를 임베딩 벡터로 변환

        Raises:
            RuntimeError: OpenAI API 호출 실패 시
        """
        text = text.replace("\n", " ")
        try:
            response = self.openai_client.embeddings.create(
                input=[text],
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"임베딩 생성 실패 ('{text[:30]}...'): {e}") from e

    def search_sign_description(self, keyword: str, limit: int = 1) -> Optional[str]:
        """
        키워드로 수어 설명을 검색합니다.

        Args:
            keyword: 검색할 단어
            limit: 반환할 결과 수

        Returns:
            가장 적합한 수어 설명 (없으면 None)

        Raises:
            RuntimeError: 임베딩 생성 또는 Qdrant 검색 실패 시
        """
        # 첫 Qdrant 접근 시점에만 데이터 유무 확인 및 자동 적재
        self._ensure_collection_populated_if_needed()

        # 1. 키워드 임베딩
        query_vector = self._get_embedding(keyword)

        # 2. 벡터 검색
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=self.score_threshold,
            )
            search_result = response.points
        except Exception as e:
            raise RuntimeError(f"Qdrant 검색 실패 ('{keyword}'): {e}") from e

        if not search_result:
            logger.info(
                f"No sign description found for '{keyword}' "
                f"(threshold={self.score_threshold})"
            )
            return None

        # 가장 높은 점수의 결과 반환
        top_result = search_result[0]
        description = top_result.payload.get("description")

        logger.info(
            f"Found match for '{keyword}': "
            f"{top_result.payload.get('keywords')} "
            f"(Score: {top_result.score:.4f}, threshold={self.score_threshold})"
        )
        return description

    def get_service_status(self) -> Dict[str, Any]:
        """서비스 상태 확인"""
        status = {
            "qdrant_connection": False,
            "openai_connection": True,
            "collection_exists": False,
            "document_count": 0,
            "score_threshold": self.score_threshold,
        }

        try:
            collections = self.client.get_collections()
            status["qdrant_connection"] = True

            # 컬렉션 존재 여부 확인
            for col in collections.collections:
                if col.name == self.collection_name:
                    status["collection_exists"] = True
                    count_result = self.client.count(
                        collection_name=self.collection_name
                    )
                    status["document_count"] = count_result.count
                    break
        except Exception as e:
            logger.warning(f"Qdrant 상태 확인 실패: {e}")

        return status
