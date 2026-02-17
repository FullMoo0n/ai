import argparse
import os
import sys
import uuid
import pandas as pd
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
from dotenv import load_dotenv

# .env 로드
load_dotenv()

# 설정
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "sign_languages"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 데이터 파일 경로
DATA_FILE = "data/문화체육관광부 국립국어원_한국수어사전_한국어대응표현정보_20240909.csv"


def init_qdrant_collection(client: QdrantClient, recreate: bool = False):
    """Qdrant 컬렉션 초기화"""
    collections = client.get_collections()
    exists = any(c.name == COLLECTION_NAME for c in collections.collections)

    if exists:
        if recreate:
            print(f"Collection '{COLLECTION_NAME}' exists. --recreate flag set, deleting...")
            client.delete_collection(COLLECTION_NAME)
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Reusing. (use --recreate to force)")
            return

    # 컬렉션 생성 (OpenAI text-embedding-3-small 차원: 1536)
    # create_collection is idempotent if it doesn't exist, but if it exists and we didn't delete it (recreate=False), this line might error if we don't check existence first.
    # checking exists again or using try-except is safer, but client.create_collection typically raises if exists.
    # However, we only reach here if (not exists) OR (exists and recreate=True which deleted it).
    # So it is safe to create.
    
    # But wait, if exists=True and recreate=False, we returned early above.
    # So we are good.
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def get_embeddings(
    client: OpenAI,
    texts: List[str],
) -> Tuple[List[List[float]], List[int]]:
    """배치로 임베딩 생성

    Returns:
        (embeddings, valid_indices): 임베딩 리스트와 원본 배치 내 유효 인덱스 리스트.
        두 리스트의 길이는 동일하며, valid_indices[k]는 embeddings[k]에 대응하는
        원본 texts 리스트 내의 인덱스입니다.
    """
    # 유효한 텍스트와 원본 인덱스를 함께 추적
    valid_indices = [
        i for i, t in enumerate(texts) if t and isinstance(t, str) and t.strip()
    ]
    valid_texts = [texts[i].replace("\n", " ") for i in valid_indices]

    if not valid_texts:
        return [], []

    try:
        response = client.embeddings.create(
            input=valid_texts,
            model="text-embedding-3-small"
        )
        embeddings = [data.embedding for data in response.data]
        return embeddings, valid_indices
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return [], []


def ingest_data(recreate_collection: bool = False):
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY is missing.")
        return

    # 클라이언트 초기화
    qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    openai = OpenAI(api_key=OPENAI_API_KEY)

    # CSV 로드
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {len(df)} rows from {DATA_FILE}")
    except FileNotFoundError:
        print(f"Error: File not found at {DATA_FILE}")
        return

    # 컬렉션 초기화
    init_qdrant_collection(qdrant, recreate=recreate_collection)

    # 배치 처리
    batch_size = 100
    total_processed = 0
    failed_batches = 0

    # NaN 처리 강화: fillna('') 후 astype(str)
    keywords = df['한국어 대응표현'].fillna('').astype(str).tolist()
    descriptions = df['수형설명'].fillna('').astype(str).tolist()
    categories = df['대/중 분류'].fillna('').astype(str).tolist()
    item_ids = df['수어 표제어 번호'].fillna('').astype(str).tolist()

    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_keywords = keywords[i:batch_end]

        # 임베딩 생성 (유효 인덱스도 함께 반환)
        vectors, valid_indices = get_embeddings(openai, batch_keywords)

        if not vectors:
            # 배치 내 유효한 텍스트가 하나도 없었던 경우 or 에러
            # 에러가 아니고 그냥 빈 텍스트들만 있었으면 failed_batches로 칠 필요는 없으나,
            # 원본 코드는 vectors가 없으면 실패로 간주했음. warning 로그만 남기고 continue.
            print(f"⚠️ Batch {i // batch_size + 1} skipped or failed (rows {i}-{batch_end - 1})")
            continue

        current_batch_points = []
        for k, vector in enumerate(vectors):
            # valid_indices[k]는 배치 내 원본 인덱스 → 전체 인덱스로 변환
            original_idx = i + valid_indices[k]

            # Point ID: UUID 사용으로 충돌 방지
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"sign-{item_ids[original_idx]}-{original_idx}"))

            payload = {
                "keywords": keywords[original_idx],
                "description": descriptions[original_idx],
                "category": categories[original_idx],
                "item_id": item_ids[original_idx],
            }

            point = models.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
            current_batch_points.append(point)

        # 업로드
        if current_batch_points:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=current_batch_points,
            )
            total_processed += len(current_batch_points)
            print(f"Processed {total_processed}/{len(df)} items...")

    # 요약 로그
    print("=" * 50)
    print(f"✅ Data ingestion completed!")
    print(f"   Total processed: {total_processed}/{len(df)}")
    if failed_batches:
        print(f"   ⚠️ Failed batches: {failed_batches}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest sign language data into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Force recreation of the Qdrant collection (DELETES EXISTING DATA)",
    )
    args = parser.parse_args()

    ingest_data(recreate_collection=args.recreate)
