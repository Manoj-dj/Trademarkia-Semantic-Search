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
    Cluster-partitioned semantic cache built from first principles.

    Architecture:
    Entries are stored in per-cluster buckets:
        _store: Dict[cluster_id (int), List[CacheEntry]]

    On lookup, we do not scan all N cache entries (O(N)). Instead:
    1. Compute the query's fuzzy membership distribution via GMM.
    2. Identify all cluster buckets where membership > multi_bucket_threshold.
    3. Search only those buckets.
    This reduces lookup complexity from O(N) to O(N/K) on average, and makes
    the cache scale sub-linearly as the number of entries grows.

    Multi-bucket lookup is used (not just argmax) because a query about
    "gun legislation" may belong 55% to politics and 35% to firearms.
    Searching only the politics bucket would miss cached entries stored
    under the firearms bucket by a prior semantically similar query.

    Similarity metric:
    Cosine similarity via dot product on L2-normalized embeddings.
    Since embeddings are unit-normalized at encode time, dot(a, b) = cos(a, b).

    Tunable parameter — similarity threshold (theta):
    This is the single most consequential design variable in the cache.
    - theta = 0.99: near-exact match required. Almost no paraphrase pairs hit.
      System degrades to FAISS search on every query. Very low hit rate.
    - theta = 0.75: catches paraphrases but also returns results for loosely
      related queries. High hit rate but low precision — wrong answers returned.
    - theta = 0.85 (default): empirically balanced. Catches clear paraphrases
      ("gun control debate" / "firearm regulation discussion") while rejecting
      queries that are semantically adjacent but not equivalent.
    The threshold is exposed as an environment variable (CACHE_SIM_THRESHOLD)
    because the optimal value is use-case dependent: a strict legal search
    system may prefer 0.93; a conversational interface may use 0.80.

    No Redis, Memcached, or external caching library is used.
    All state is held in Python in-process memory.
    """

    _instance: Optional["SemanticCache"] = None

    def __new__(cls, *args, **kwargs):
        # Singleton: one cache instance per server process
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        sim_threshold: float = 0.85,
        multi_bucket_threshold: float = 0.2,
    ) -> None:
        # Guard against repeated __init__ calls on the same Singleton instance
        if getattr(self, "_initialized", False):
            return

        self._store: Dict[int, List[CacheEntry]] = {}
        self._sim_threshold = sim_threshold
        self._multi_bucket_threshold = multi_bucket_threshold
        self._hit_count = 0
        self._miss_count = 0
        self._initialized = True

        logger.info(
            "SemanticCache initialized | sim_threshold=%.2f | multi_bucket_threshold=%.2f",
            self._sim_threshold,
            self._multi_bucket_threshold,
        )

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

        Returns:
            (cache_hit: bool, matched_entry: Optional[CacheEntry], best_score: float)
        """
        # Identify all active cluster buckets (multi-bucket strategy)
        active_clusters = [
            int(i)
            for i, prob in enumerate(cluster_probs)
            if float(prob) > self._multi_bucket_threshold
        ]
        if not active_clusters:
            active_clusters = [int(np.argmax(cluster_probs))]

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None

        for cluster_id in active_clusters:
            bucket = self._store.get(cluster_id, [])
            for entry in bucket:
                score = self._cosine_similarity(query_embedding, entry.query_embedding)
                if score > best_score:
                    best_score = score
                    best_entry = entry

        if best_score >= self._sim_threshold and best_entry is not None:
            self._hit_count += 1
            logger.info(
                "Cache HIT | score=%.4f | threshold=%.2f | matched='%s'",
                best_score,
                self._sim_threshold,
                best_entry.query_text[:60],
            )
            return True, best_entry, best_score

        self._miss_count += 1
        logger.info(
            "Cache MISS | best_score=%.4f | threshold=%.2f",
            best_score,
            self._sim_threshold,
        )
        return False, None, best_score

    def store(
        self,
        query_text: str,
        query_embedding: np.ndarray,
        result: list,
        cluster_probs: np.ndarray,
    ) -> None:
        """
        Store a new entry in the dominant cluster bucket.
        Dominant cluster (argmax) is used for storage to ensure
        deterministic, single-bucket assignment per entry.
        """
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
            "Cache STORE | cluster=%d | query='%s' | bucket_size=%d",
            dominant_cluster,
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
