"""
SignDataService 유닛 테스트

테스트 항목:
- 임베딩 생성
- 벡터 검색 기능
- Qdrant/OpenAI 클라이언트 사용 불가 시 오류 처리
- 서비스 상태 확인 (get_service_status)
"""

import os
import unittest
from unittest.mock import patch, MagicMock, PropertyMock


class TestSignDataServiceInit(unittest.TestCase):
    """초기화 관련 테스트"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_init_success(self, mock_openai, mock_qdrant):
        """정상 초기화"""
        from app.services.sign_data_service import SignDataService
        service = SignDataService()
        self.assertIsNotNone(service.client)
        self.assertIsNotNone(service.openai_client)
        self.assertEqual(service.score_threshold, 0.7)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient", side_effect=Exception("connection refused"))
    def test_init_qdrant_failure_raises(self, mock_qdrant):
        """Qdrant 연결 실패 시 RuntimeError 발생"""
        from app.services.sign_data_service import SignDataService
        with self.assertRaises(RuntimeError) as ctx:
            SignDataService()
        self.assertIn("Qdrant 클라이언트 초기화 실패", str(ctx.exception))

    @patch.dict(os.environ, {"QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"}, clear=True)
    @patch("app.services.sign_data_service.QdrantClient")
    def test_init_missing_openai_key_raises(self, mock_qdrant):
        """OPENAI_API_KEY 미설정 시 RuntimeError 발생"""
        # clear=True 로 OPENAI_API_KEY 제거
        from app.services.sign_data_service import SignDataService
        with self.assertRaises(RuntimeError) as ctx:
            SignDataService()
        self.assertIn("OPENAI_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {
        "OPENAI_API_KEY": "test-key",
        "QDRANT_HOST": "localhost",
        "QDRANT_PORT": "6333",
        "QDRANT_SCORE_THRESHOLD": "0.5",
    })
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_init_custom_score_threshold_from_env(self, mock_openai, mock_qdrant):
        """환경변수로 score_threshold 설정"""
        from app.services.sign_data_service import SignDataService
        service = SignDataService()
        self.assertEqual(service.score_threshold, 0.5)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_init_custom_score_threshold_from_param(self, mock_openai, mock_qdrant):
        """파라미터로 score_threshold 설정 (환경변수보다 우선)"""
        from app.services.sign_data_service import SignDataService
        service = SignDataService(score_threshold=0.9)
        self.assertEqual(service.score_threshold, 0.9)


class TestSignDataServiceEmbedding(unittest.TestCase):
    """임베딩 생성 테스트"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_get_embedding_success(self, mock_openai_cls, mock_qdrant):
        """임베딩 정상 생성"""
        from app.services.sign_data_service import SignDataService

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_openai_cls.return_value.embeddings.create.return_value = mock_response

        service = SignDataService()
        result = service._get_embedding("테스트")
        self.assertEqual(len(result), 1536)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_get_embedding_api_failure_raises(self, mock_openai_cls, mock_qdrant):
        """OpenAI API 호출 실패 시 RuntimeError 발생"""
        from app.services.sign_data_service import SignDataService

        mock_openai_cls.return_value.embeddings.create.side_effect = Exception("API Error")

        service = SignDataService()
        with self.assertRaises(RuntimeError) as ctx:
            service._get_embedding("테스트")
        self.assertIn("임베딩 생성 실패", str(ctx.exception))


class TestSignDataServiceSearch(unittest.TestCase):
    """벡터 검색 테스트"""

    def _create_service(self, mock_openai_cls, mock_qdrant):
        """테스트용 서비스 인스턴스 생성 헬퍼"""
        from app.services.sign_data_service import SignDataService

        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1] * 1536
        mock_response = MagicMock()
        mock_response.data = [mock_embedding]
        mock_openai_cls.return_value.embeddings.create.return_value = mock_response

        return SignDataService()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_search_returns_description(self, mock_openai_cls, mock_qdrant):
        """검색 성공 시 수어 설명 반환"""
        service = self._create_service(mock_openai_cls, mock_qdrant)

        mock_point = MagicMock()
        mock_point.payload = {"description": "두 손을 펴서...", "keywords": "공주"}
        mock_point.score = 0.95
        mock_query_response = MagicMock()
        mock_query_response.points = [mock_point]
        service.client.query_points.return_value = mock_query_response

        result = service.search_sign_description("공주")
        self.assertEqual(result, "두 손을 펴서...")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_search_no_results_returns_none(self, mock_openai_cls, mock_qdrant):
        """검색 결과 없을 때 None 반환"""
        service = self._create_service(mock_openai_cls, mock_qdrant)

        mock_query_response = MagicMock()
        mock_query_response.points = []
        service.client.query_points.return_value = mock_query_response

        result = service.search_sign_description("없는단어")
        self.assertIsNone(result)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_search_qdrant_failure_raises(self, mock_openai_cls, mock_qdrant):
        """Qdrant 검색 실패 시 RuntimeError 발생"""
        service = self._create_service(mock_openai_cls, mock_qdrant)
        service.client.query_points.side_effect = Exception("timeout")

        with self.assertRaises(RuntimeError) as ctx:
            service.search_sign_description("테스트")
        self.assertIn("Qdrant 검색 실패", str(ctx.exception))


class TestSignDataServiceStatus(unittest.TestCase):
    """서비스 상태 확인 테스트"""

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_get_service_status_healthy(self, mock_openai_cls, mock_qdrant):
        """정상 상태 확인"""
        from app.services.sign_data_service import SignDataService
        service = SignDataService()

        mock_col = MagicMock()
        mock_col.name = "sign_languages"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_col]
        service.client.get_collections.return_value = mock_collections

        mock_count = MagicMock()
        mock_count.count = 13950
        service.client.count.return_value = mock_count

        status = service.get_service_status()
        self.assertTrue(status["qdrant_connection"])
        self.assertTrue(status["openai_connection"])
        self.assertTrue(status["collection_exists"])
        self.assertEqual(status["document_count"], 13950)
        self.assertEqual(status["score_threshold"], 0.7)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "QDRANT_HOST": "localhost", "QDRANT_PORT": "6333"})
    @patch("app.services.sign_data_service.QdrantClient")
    @patch("app.services.sign_data_service.OpenAI")
    def test_get_service_status_qdrant_error(self, mock_openai_cls, mock_qdrant):
        """Qdrant 연결 실패 시 상태"""
        from app.services.sign_data_service import SignDataService
        service = SignDataService()
        service.client.get_collections.side_effect = Exception("connection lost")

        status = service.get_service_status()
        self.assertFalse(status["qdrant_connection"])


if __name__ == "__main__":
    unittest.main()
