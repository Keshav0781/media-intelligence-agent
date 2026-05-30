import os
import pickle
import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector, Prefetch, FusionQuery, Fusion
)
from rank_bm25 import BM25Okapi

# Suppress HuggingFace warning
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

# Qdrant local storage path
QDRANT_PATH = "data/qdrant"

# BM25 model cache directory
BM25_CACHE_DIR = "data/bm25"

# Embedding model — multilingual, handles German + English queries
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Vector dimension for this model
VECTOR_DIM = 384

# Singleton instances — created once, reused throughout
_embedding_model = None
_qdrant_client = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load embedding model — singleton pattern.
    Model loaded once and reused for all subsequent calls.
    Avoids reloading 471MB model on every function call.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    """
    Get Qdrant client — singleton pattern.
    Single client instance shared across all operations.
    Qdrant local mode allows only one client at a time per folder.
    """
    global _qdrant_client
    if _qdrant_client is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def get_collection_name(video_hash: str) -> str:
    """
    Generate collection name from video hash.
    Each video gets its own collection in Qdrant.
    """
    return f"video_{video_hash}"


def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    """
    Create Qdrant collection with both dense and sparse vectors.
    Dense vectors — for semantic/natural language search.
    Sparse vectors — for keyword/BM25 search.
    Combined — hybrid search handles both query styles.
    """
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=False)
                )
            }
        )
        print(f"Created hybrid collection: {collection_name}")
    else:
        print(f"Collection already exists: {collection_name}")


def build_segment_context(segments: List[Dict], index: int) -> str:
    """
    Build overlapping context for a segment.
    Includes previous, current and next segment text.
    This ensures search finds content even at segment boundaries.
    """
    parts = []

    if index > 0:
        parts.append(segments[index - 1]["text"].strip())

    parts.append(segments[index]["text"].strip())

    if index < len(segments) - 1:
        parts.append(segments[index + 1]["text"].strip())

    return " ".join(parts)


def index_video(
    video_hash: str,
    video_path: str,
    transcript: Dict,
    analysis: Dict
) -> int:
    """
    Index all video content in Qdrant using hybrid search.

    Uses:
    - Dense vectors (semantic) — for natural language queries
    - Sparse vectors (BM25) — for keyword queries
    - Batch encoding — all texts encoded in one call
    - Overlap context — neighbouring segments included
    - BM25 model persisted to disk for accurate query time search

    Indexes:
    1. Whisper segments with overlap context
    2. Key stories from Gemini analysis
    3. Overall summary
    4. Main topics
    """
    model = get_embedding_model()
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)

    create_collection_if_not_exists(client, collection_name)

    segments = transcript["segments"]

    # --- STEP 1: Collect all texts and metadata ---
    all_texts = []
    all_payloads = []

    # 1. Collect segment texts with overlap context
    print(f"Collecting {len(segments)} transcript segments...")
    for i, segment in enumerate(segments):
        context_text = build_segment_context(segments, i)
        all_texts.append(context_text)
        all_payloads.append({
            "text": context_text,
            "original_text": segment["text"].strip(),
            "type": "segment",
            "timestamp_start": segment["start"],
            "timestamp_end": segment["end"],
            "video_path": video_path,
            "language": transcript["language"]
        })

    # 2. Collect key stories
    print(f"Collecting {len(analysis.get('key_stories', []))} key stories...")
    for story in analysis.get("key_stories", []):
        all_texts.append(story["story"])
        all_payloads.append({
            "text": story["story"],
            "original_text": story["story"],
            "type": "story",
            "timestamp_start": story["timestamp"],
            "timestamp_end": story["timestamp"],
            "video_path": video_path,
            "language": analysis.get("language", "unknown")
        })

    # 3. Collect overall summary
    summary_text = analysis.get("overall_summary", "")
    if summary_text:
        print("Collecting overall summary...")
        all_texts.append(summary_text)
        all_payloads.append({
            "text": summary_text,
            "original_text": summary_text,
            "type": "summary",
            "timestamp_start": 0,
            "timestamp_end": 0,
            "video_path": video_path,
            "language": analysis.get("language", "unknown")
        })

    # 4. Collect main topics
    print(f"Collecting {len(analysis.get('main_topics', []))} main topics...")
    for topic in analysis.get("main_topics", []):
        all_texts.append(topic)
        all_payloads.append({
            "text": topic,
            "original_text": topic,
            "type": "topic",
            "timestamp_start": 0,
            "timestamp_end": 0,
            "video_path": video_path,
            "language": analysis.get("language", "unknown")
        })

    # --- STEP 2: Build BM25 index and persist to disk ---
    print("Building BM25 index for keyword search...")
    tokenized_corpus = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    # Persist BM25 model for accurate query time sparse vectors
    os.makedirs(BM25_CACHE_DIR, exist_ok=True)
    bm25_path = os.path.join(BM25_CACHE_DIR, f"{collection_name}.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    print(f"BM25 model saved: {bm25_path}")

    # --- STEP 3: Batch encode all texts for dense vectors ---
    print(f"Batch encoding {len(all_texts)} texts...")
    dense_embeddings = model.encode(all_texts)

    # --- STEP 4: Build Qdrant points with both dense and sparse vectors ---
    points = []
    for i, (text, dense_emb, payload) in enumerate(zip(all_texts, dense_embeddings, all_payloads)):
        # Build sparse vector using persisted BM25
        tokens = text.lower().split()
        scores = bm25.get_scores(tokens)
        indices = np.where(scores > 0)[0].tolist()
        values = scores[indices].tolist()
        sparse_vec = SparseVector(indices=indices, values=values)

        points.append(PointStruct(
            id=i + 1,
            vector={
                "dense": dense_emb.tolist(),
                "sparse": sparse_vec
            },
            payload=payload
        ))

    # --- STEP 5: Store all points in Qdrant ---
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    total = len(points)
    print(f" Indexed {total} documents in collection '{collection_name}'")
    return total


def search(
    video_hash: str,
    query: str,
    limit: int = 5,
    content_type: str = None
) -> List[Dict]:
    """
    Hybrid search — combines semantic and keyword search using RRF.

    Handles all query styles:
    - Natural language: "minister who lied about toll road"
    - Keywords: "Scheuer PKW Maut"
    - Mixed language: "Scheuer court case"

    Uses persisted BM25 model for accurate sparse query vectors.

    Args:
        video_hash: Hash of video to search in
        query: Natural language or keyword search query
        limit: Maximum number of results to return
        content_type: Optional filter - 'segment', 'story', 'summary', 'topic'

    Returns:
        List of relevant results with text, timestamp and score
    """
    model = get_embedding_model()
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)

    if not client.collection_exists(collection_name):
        raise ValueError(f"No index found for video hash: {video_hash}")

    # Dense query vector — semantic search
    query_dense = model.encode(query).tolist()

    # Load persisted BM25 model for accurate sparse query vector
    bm25_path = os.path.join(BM25_CACHE_DIR, f"{collection_name}.pkl")
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        query_tokens = query.lower().split()
        query_scores = bm25.get_scores(query_tokens)
        query_indices = np.where(query_scores > 0)[0].tolist()
        query_values = query_scores[query_indices].tolist()
        query_sparse = SparseVector(indices=query_indices, values=query_values)
    else:
        # Fallback — dense search only if BM25 not found
        print("Warning: BM25 model not found, using dense search only")
        query_sparse = SparseVector(indices=[], values=[])

    # Build filter if content_type specified
    query_filter = None
    if content_type:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="type",
                    match=MatchValue(value=content_type)
                )
            ]
        )

    # Hybrid search using RRF fusion
    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            Prefetch(
                query=query_dense,
                using="dense",
                limit=limit * 2,
                filter=query_filter
            ),
            Prefetch(
                query=query_sparse,
                using="sparse",
                limit=limit * 2,
                filter=query_filter
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        query_filter=query_filter
    )

    formatted = []
    for point in results.points:
        formatted.append({
            "text": point.payload["original_text"],
            "context": point.payload["text"],
            "type": point.payload["type"],
            "timestamp_start": point.payload["timestamp_start"],
            "timestamp_end": point.payload.get("timestamp_end", ""),
            "score": point.score,
            "video_path": point.payload["video_path"]
        })

    return formatted