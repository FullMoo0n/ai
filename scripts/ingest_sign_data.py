import os
import sys
import pandas as pd
from typing import List
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

def init_qdrant_collection(client: QdrantClient):
    """Qdrant 컬렉션 초기화"""
    collections = client.get_collections()
    exists = any(c.name == COLLECTION_NAME for c in collections.collections)
    
    if exists:
        print(f"Collection '{COLLECTION_NAME}' already exists. Recreating...")
        client.delete_collection(COLLECTION_NAME)
    
    # 컬렉션 생성 (OpenAI text-embedding-3-small 차원: 1536)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=1536,
            distance=models.Distance.COSINE
        )
    )
    print(f"Collection '{COLLECTION_NAME}' created.")

def get_embeddings(client: OpenAI, texts: List[str]) -> List[List[float]]:
    """배치로 임베딩 생성"""
    # 빈 문자열이나 None 제거 및 전처리
    valid_indices = [i for i, t in enumerate(texts) if t and isinstance(t, str) and t.strip()]
    valid_texts = [texts[i].replace("\n", " ") for i in valid_indices]
    
    if not valid_texts:
        return []

    try:
        response = client.embeddings.create(
            input=valid_texts,
            model="text-embedding-3-small"
        )
        # 원래 순서대로 매핑 (유효하지 않은 값은 None 처리 등 필요하지만 여기선 단순화)
        embeddings = [data.embedding for data in response.data]
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def ingest_data():
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
    init_qdrant_collection(qdrant)
    
    # 배치 처리
    batch_size = 100
    total_processed = 0
    
    # 데이터프레임 순회
    # 필요한 컬럼: '한국어 대응표현' (임베딩 대상), '수형설명' (메타데이터), '대/중 분류' (메타데이터)
    # 컬럼명이 정확한지 확인 필요
    
    keywords = df['한국어 대응표현'].astype(str).tolist()
    descriptions = df['수형설명'].astype(str).tolist()
    categories = df['대/중 분류'].astype(str).tolist()
    item_ids = df['수어 표제어 번호'].astype(str).tolist()

    points = []
    
    for i in range(0, len(df), batch_size):
        batch_end = min(i + batch_size, len(df))
        batch_keywords = keywords[i:batch_end]
        
        # 임베딩 생성
        vectors = get_embeddings(openai, batch_keywords)
        
        if not vectors:
            continue
            
        current_batch_points = []
        for j, vector in enumerate(vectors):
            idx = i + j
            
            # Payload 구성
            payload = {
                "keywords": keywords[idx],
                "description": descriptions[idx],
                "category": categories[idx],
                "item_id": item_ids[idx]
            }
            
            point = models.PointStruct(
                id=int(item_ids[idx]) if item_ids[idx].isdigit() else idx, # ID는 정수 권장
                vector=vector,
                payload=payload
            )
            current_batch_points.append(point)
            
        # 업로드
        if current_batch_points:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=current_batch_points
            )
            total_processed += len(current_batch_points)
            print(f"Processed {total_processed}/{len(df)} items...")

    print("Data ingestion completed successfully!")

if __name__ == "__main__":
    ingest_data()
