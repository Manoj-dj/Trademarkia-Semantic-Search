import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from logger.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    query_text: str
    query_embedding: np.ndarray
    result: list
    dominant_cluster: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


class SemanticCache:
    """
    Cluster-partitioned semantic cache with adaptive per-cluster thresholds.

    Architecture:
    Entries are stored in per-cluster buckets:
        _store: Dict[cluster_id (int), List[CacheEntry]]

    Lookup complexity: O(N/K) — only active cluster buckets are searched.

    Threshold strategy (adaptive):
    Each cluster has its own similarity threshold derived from the intra-cluster
    pairwise cosine similarity distribution of its member documents:

        theta_k = clip(mean_k + alpha * std_k, 0.60, 0.92)

    Dense clusters (tight UMAP blob, high mean_k) → higher theta → stricter matching.
    Sparse clusters (wide spread, low mean_k) → lower theta → looser matching.

    This is fundamentally more correct than a global threshold because it
    calibrates the definition of "similar enough" to the local geometry of
    each semantic cluster. A global threshold either over-matches in dense
    clusters (false hits) or under-matches in sparse clusters (missed hits).

    The global threshold (CACHE_SIM_THRESHOLD env var) is retained as the
    fallback for any cluster not covered by the adaptive thresholds dict.

    No Redis, Memcached, or external caching library is used.
    """

    _instance: Optional["SemanticCache"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        sim_threshold: float = 0.70,
        multi_bucket_threshold: float = 0.2,
        per_cluster_thresholds: Optional[Dict[int, float]] = None,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self._store: Dict[int, List[CacheEntry]] = {}
        self._sim_threshold = sim_threshold
        self._multi_bucket_threshold = multi_bucket_threshold

        # Adaptive per-cluster thresholds — falls back to global if cluster not found
        self._per_cluster_thresholds: Dict[int, float] = per_cluster_thresholds or {}

        self._hit_count = 0
        self._miss_count = 0
        self._initialized = True

        logger.info(
            "SemanticCache initialized | global_threshold=%.2f | "
            "multi_bucket_threshold=%.2f | adaptive_clusters=%d",
            self._sim_threshold,
            self._multi_bucket_threshold,
            len(self._per_cluster_thresholds),
        )
        if self._per_cluster_thresholds:
            logger.info("Per-cluster thresholds: %s", self._per_cluster_thresholds)

    def _resolve_threshold(self, cluster_id: int) -> float:
        """
        Return the adaptive threshold for a cluster if available,
        otherwise fall back to the global threshold.
        """
        return self._per_cluster_thresholds.get(cluster_id, self._sim_threshold)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Embeddings are L2-normalized at encoding time.
        # Dot product of unit vectors equals cosine similarity.
        return float(np.dot(a.flatten(), b.flatten()))

    def lookup(
        self,
        query_embedding: np.ndarray,
        cluster_probs: np.ndarray,
    ) -> Tuple[bool, Optional[CacheEntry], float]:
        """
        Search the cache for a semantically equivalent prior query.

        For each active cluster bucket, the threshold used is the
        cluster-specific adaptive threshold — not a single global value.
        This ensures matching criteria reflect the local semantic density
        of each cluster.

        Returns: (cache_hit, matched_entry, best_score)
        """
        active_clusters = [
            int(i)
            for i, prob in enumerate(cluster_probs)
            if float(prob) > self._multi_bucket_threshold
        ]
        if not active_clusters:
            active_clusters = [int(np.argmax(cluster_probs))]

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None
        best_cluster: int = -1

        for cluster_id in active_clusters:
            bucket = self._store.get(cluster_id, [])
            for entry in bucket:
                score = self._cosine_similarity(query_embedding, entry.query_embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry
                    best_cluster = cluster_id

        # Resolve the threshold for the cluster that produced the best match
        effective_threshold = (
            self._resolve_threshold(best_cluster)
            if best_cluster >= 0
            else self._sim_threshold
        )

        logger.debug(
            "Cache lookup | best_cluster=%d | adaptive_theta=%.4f | best_score=%.4f",
            best_cluster, effective_threshold, best_score
        )

        if best_score >= effective_threshold and best_entry is not None:
            self._hit_count += 1
            logger.info(
                "Cache HIT | score=%.4f | adaptive_theta=%.4f | cluster=%d | matched='%s'",
                best_score,
                effective_threshold,
                best_cluster,
                best_entry.query_text[:60],
            )
            return True, best_entry, best_score

        self._miss_count += 1
        logger.info(
            "Cache MISS | best_score=%.4f | effective_theta=%.4f | cluster=%d",
            best_score, effective_threshold, best_cluster
        )
        return False, None, best_score

    def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: list,
        cluster_probs: np.ndarray,
    ) -> None:
        dominant_cluster = int(np.argmax(cluster_probs))
        entry = CacheEntry(
            query_text=query_text,
            query_embedding=query_embedding.copy(),
            result=result,
            dominant_cluster=dominant_cluster,
        )
        if dominant_cluster not in self._store:
            self._store[dominant_cluster] = []
        self._store[dominant_cluster].append(entry)
        logger.info(
            "Cache STORE | cluster=%d | adaptive_theta=%.4f | query='%s' | bucket_size=%d",
            dominant_cluster,
            self._resolve_threshold(dominant_cluster),
            query_text[:60],
            len(self._store[dominant_cluster]),
        )

    def get_stats(self) -> Dict:
        total_entries = sum(len(v) for v in self._store.values())
        total_queries = self._hit_count + self._miss_count
        hit_rate = (
            round(self._hit_count / total_queries, 4) if total_queries > 0 else 0.0
        )
        return {
            "total_entries": total_entries,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
        }

    def clear(self) -> int:
        total_entries = sum(len(v) for v in self._store.values())
        self._store.clear()
        self._hit_count = 0
        self._miss_count = 0
        logger.info(
            "Cache cleared | removed_entries=%d | stats_reset=True", total_entries
        )
        return total_entries
