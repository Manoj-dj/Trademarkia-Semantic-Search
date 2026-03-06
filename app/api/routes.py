import numpy as np
from fastapi import APIRouter, Request, HTTPException

from app.models import (
    QueryRequest,
    QueryResponse,
    CacheStatsResponse,
    DeleteCacheResponse,
    DocumentResult,
    HealthResponse,
)
from app.pipeline.indexer import search_index
from app.pipeline.clusterer import predict_query_membership
from logger.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Liveness check. Confirms that the FAISS index and GMM model
    are loaded and the cache is operational.
    """
    state = request.app.state
    return HealthResponse(
        status="ok",
        index_loaded=getattr(state, "faiss_index", None) is not None,
        cluster_model_loaded=getattr(state, "gmm", None) is not None,
        cache_entries=state.cache.get_stats()["total_entries"],
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: Request, body: QueryRequest):
    """
    Semantic search endpoint.

    Pipeline:
    1. Validate and embed the incoming query.
    2. Compute fuzzy cluster membership (GMM predict_proba via PCA projection).
    3. Search semantic cache using cluster-partitioned cosine lookup.
       - Hit: return cached result immediately, no FAISS call.
       - Miss: search FAISS, construct result, store in cache, return result.

    The dominant cluster returned in the response is the argmax of the query's
    membership distribution — the cluster it most strongly belongs to.
    """
    state = request.app.state
    query_text = body.query.strip()

    if not query_text:
        raise HTTPException(status_code=400, detail="Query string cannot be empty")

    logger.info("Query received | text='%s'", query_text[:80])

    # Step 1: Embed query into L2-normalized vector
    query_embedding = state.embedder.encode_query(query_text)

    # Step 2: Determine fuzzy cluster membership
    cluster_probs = predict_query_membership(state.gmm, state.pca, query_embedding)
    dominant_cluster = int(np.argmax(cluster_probs))
    logger.debug(
        "Cluster detection | dominant=%d | prob=%.3f",
        dominant_cluster,
        float(cluster_probs[dominant_cluster]),
    )

    # Step 3: Cache lookup
    cache_hit, matched_entry, similarity_score = state.cache.lookup(
        query_embedding, cluster_probs
    )

    if cache_hit and matched_entry is not None:
        return QueryResponse(
            query=query_text,
            cache_hit=True,
            matched_query=matched_entry.query_text,
            similarity_score=round(float(similarity_score), 4),
            result=[DocumentResult(**doc) for doc in matched_entry.result],
            dominant_cluster=dominant_cluster,
        )

    # Step 4: Cache miss — search FAISS index
    logger.info("Cache miss | performing FAISS search | query='%s'", query_text[:80])
    raw_results = search_index(
        index=state.faiss_index,
        query_embedding=query_embedding,
        metadata_map=state.metadata_map,
        top_k=state.settings.top_k_results,
    )

    # Step 5: Store result in cache for future reuse
    state.cache.store(
        query_text=query_text,
        query_embedding=query_embedding,
        result=raw_results,
        cluster_probs=cluster_probs,
    )

    return QueryResponse(
        query=query_text,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=[DocumentResult(**doc) for doc in raw_results],
        dominant_cluster=dominant_cluster,
    )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def cache_stats(request: Request):
    """
    Returns current cache state: total entries, hit count, miss count, hit rate.
    """
    stats = request.app.state.cache.get_stats()
    logger.info("Cache stats requested | %s", stats)
    return CacheStatsResponse(**stats)


@router.delete("/cache", response_model=DeleteCacheResponse)
async def delete_cache(request: Request):
    """
    Flush all cache entries and reset all statistics to zero.
    """
    entries_cleared = request.app.state.cache.clear()
    logger.info("Cache flushed via DELETE /cache | entries_cleared=%d", entries_cleared)
    return DeleteCacheResponse(
        message="Cache flushed and all statistics reset",
        entries_cleared=entries_cleared,
    )
