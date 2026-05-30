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
from app.logger import get_logger

# Suppress HuggingFace warning
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"

logger = get_logger(__name__)

from app.config import get_storage_config, get_rag_config
_storage_cfg = get_storage_config()
_rag_cfg = get_rag_config()

QDRANT_PATH = _storage_cfg["qdrant_path"]
BM25_CACHE_DIR = _storage_cfg["bm25_cache"]
EMBEDDING_MODEL = _rag_cfg["embedding_model"]
VECTOR_DIM = _rag_cfg["vector_dim"]

# Singleton instances — created once, reused throughout
_embedding_model = None
_qdrant_client = None


def get_embedding_model() -> SentenceTransformer:
    """
    Load embedding model — singleton pattern.
    Model loaded once and reused for all subsequent calls.
    """
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    """
    Get Qdrant client — singleton pattern.
    Single client instance shared across all operations.
    """
    global _qdrant_client
    if _qdrant_client is None:
        os.makedirs(QDRANT_PATH, exist_ok=True)
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        logger.info(f"Qdrant client initialized: {QDRANT_PATH}")
    return _qdrant_client


def get_collection_name(video_hash: str) -> str:
    """Generate collection name from video hash."""
    return f"video_{video_hash}"


def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    """Create Qdrant collection with both dense and sparse vectors."""
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
        logger.info(f"Created hybrid collection: {collection_name}")
    else:
        logger.info(f"Collection already exists: {collection_name}")


def build_segment_context(segments: List[Dict], index: int) -> str:
    """
    Build overlapping context for a segment.
    Includes previous, current and next segment text.
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
    """
    logger.info(f"Starting video indexing: {video_path}")

    model = get_embedding_model()
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)

    create_collection_if_not_exists(client, collection_name)

    segments = transcript["segments"]

    # --- STEP 1: Collect all texts and metadata ---
    all_texts = []
    all_payloads = []

    logger.info(f"Collecting {len(segments)} transcript segments...")
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

    logger.info(f"Collecting {len(analysis.get('key_stories', []))} key stories...")
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

    summary_text = analysis.get("overall_summary", "")
    if summary_text:
        logger.info("Collecting overall summary...")
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

    logger.info(f"Collecting {len(analysis.get('main_topics', []))} main topics...")
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

    # --- STEP 2: Build BM25 index and persist ---
    logger.info("Building BM25 index for keyword search...")
    tokenized_corpus = [text.lower().split() for text in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    os.makedirs(BM25_CACHE_DIR, exist_ok=True)
    bm25_path = os.path.join(BM25_CACHE_DIR, f"{collection_name}.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    logger.info(f"BM25 model saved: {bm25_path}")

    # --- STEP 3: Batch encode all texts ---
    logger.info(f"Batch encoding {len(all_texts)} texts...")
    dense_embeddings = model.encode(all_texts)

    # --- STEP 4: Build Qdrant points ---
    points = []
    for i, (text, dense_emb, payload) in enumerate(zip(all_texts, dense_embeddings, all_payloads)):
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

    # --- STEP 5: Store all points ---
    client.upsert(collection_name=collection_name, points=points)

    total = len(points)
    logger.info(f"Indexed {total} documents in collection '{collection_name}'")
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
    """
    logger.info(f"Searching: '{query}' (limit={limit}, type={content_type})")

    model = get_embedding_model()
    client = get_qdrant_client()
    collection_name = get_collection_name(video_hash)

    if not client.collection_exists(collection_name):
        logger.error(f"No index found for video hash: {video_hash}")
        raise ValueError(f"No index found for video hash: {video_hash}")

    query_dense = model.encode(query).tolist()

    bm25_path = os.path.join(BM25_CACHE_DIR, f"{collection_name}.pkl")
    if os.path.exists(bm25_path):
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        query_tokens = query.lower().split()
        query_scores = bm25.get_scores(query_tokens)
        query_indices = np.where(query_scores > 0)[0].tolist()
        query_values = query_scores[query_indices].tolist()
        query_sparse = SparseVector(indices=query_indices, values=query_values)
        logger.debug(f"BM25 sparse vector: {len(query_indices)} non-zero terms")
    else:
        logger.warning("BM25 model not found, using dense search only")
        query_sparse = SparseVector(indices=[], values=[])

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

    logger.info(f"Search returned {len(formatted)} results")
    return formatted