import json
import numpy as np
import faiss
from pathlib import Path
from typing import Dict, List, Any

from logger.logging_config import get_logger

logger = get_logger(__name__)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a FAISS IndexFlatIP over the L2-normalized corpus embeddings.

    Index type rationale:
    IndexFlatIP performs exact exhaustive inner product search. For L2-normalized
    vectors, inner product equals cosine similarity. We choose exact search
    (no approximation) because the corpus is ~20K documents — a dataset size
    where exhaustive search completes in 2–5ms, making approximate methods
    (IVFPQ, HNSW) unnecessary overhead with no practical latency benefit.
    """
    dimension = embeddings.shape[1]
    logger.info(
        "Building FAISS IndexFlatIP | dimension=%d | vectors=%d",
        dimension,
        len(embeddings),
    )
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    logger.info("FAISS index built | total_vectors=%d", index.ntotal)
    return index


def save_faiss_index(index: faiss.IndexFlatIP, path: Path) -> None:
    logger.info("Saving FAISS index to %s", path)
    faiss.write_index(index, str(path))
    logger.info("FAISS index saved")


def load_faiss_index(path: Path) -> faiss.IndexFlatIP:
    logger.info("Loading FAISS index from %s", path)
    index = faiss.read_index(str(path))
    logger.info("FAISS index loaded | total_vectors=%d", index.ntotal)
    return index


def build_metadata_map(
    documents: List[Dict],
    membership_matrix: np.ndarray,
) -> Dict[str, Any]:
    """
    Build a {faiss_id (str) -> document_info} mapping.

    FAISS stores only raw vectors and returns integer indices on search.
    This mapping bridges those indices back to the original document text,
    category label, and fuzzy cluster distribution — all required fields
    for constructing the API response.
    """
    logger.info("Building metadata map for %d documents", len(documents))
    metadata_map: Dict[str, Any] = {}

    for faiss_id, (doc, membership) in enumerate(zip(documents, membership_matrix)):
        metadata_map[str(faiss_id)] = {
            "text": doc["text"],
            "category": doc["category"],
            "word_count": doc["word_count"],
            "cluster_distribution": membership.tolist(),
        }

    logger.info("Metadata map built | entries=%d", len(metadata_map))
    return metadata_map


def save_metadata_map(metadata_map: Dict, path: Path) -> None:
    logger.info("Saving metadata map to %s", path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata_map, fh, ensure_ascii=False)
    logger.info("Metadata map saved")


def load_metadata_map(path: Path) -> Dict:
    logger.info("Loading metadata map from %s", path)
    with open(path, "r", encoding="utf-8") as fh:
        metadata_map = json.load(fh)
    logger.info("Metadata map loaded | entries=%d", len(metadata_map))
    return metadata_map


def search_index(
    index: faiss.IndexFlatIP,
    query_embedding: np.ndarray,
    metadata_map: Dict,
    top_k: int = 5,
) -> List[Dict]:
    """
    Search FAISS for the top-k most similar documents to the query embedding.
    Returns a list of result dicts suitable for constructing DocumentResult objects.
    """
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx == -1:
            continue
        meta = metadata_map.get(str(idx), {})
        results.append(
            {
                "rank": rank + 1,
                "text": meta.get("text", ""),
                "category": meta.get("category", "unknown"),
                "similarity_score": float(score),
            }
        )
    return results
