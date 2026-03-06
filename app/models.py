from typing import List, Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class DocumentResult(BaseModel):
    rank: int
    text: str
    category: str
    similarity_score: float


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str] = None
    similarity_score: Optional[float] = None
    result: List[DocumentResult]
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class DeleteCacheResponse(BaseModel):
    message: str
    entries_cleared: int


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    cluster_model_loaded: bool
    cache_entries: int
