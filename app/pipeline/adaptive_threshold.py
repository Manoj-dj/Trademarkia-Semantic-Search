import json
import numpy as np
from pathlib import Path
from typing import Dict

from logger.logging_config import get_logger

logger = get_logger(__name__)

# Sensitivity multiplier: threshold_k = mean_k + ALPHA * std_k
# Higher alpha = stricter thresholds across all clusters
_ALPHA = 0.5

# Hard bounds: prevent degenerate thresholds from extreme cluster shapes
_LOWER_BOUND = 0.60
_UPPER_BOUND = 0.92

# Maximum documents sampled per cluster for pairwise computation
# Full pairwise on 16K docs = O(N^2) — sampling keeps it tractable
_MAX_SAMPLE_PER_CLUSTER = 300


def compute_per_cluster_thresholds(
    embeddings: np.ndarray,
    membership_matrix: np.ndarray,
    alpha: float = _ALPHA,
) -> Dict[int, float]:
    """
    Compute a per-cluster adaptive similarity threshold.

    Design rationale:
    A single global threshold assumes all clusters have the same semantic
    density — this is demonstrably false. The UMAP shows Cluster 0 as a
    tight, isolated blob (high intra-similarity) while Clusters 2/3/4 are
    spread and overlapping (low intra-similarity).

    A dense cluster requires a stricter threshold: its documents are
    already close together, so a cache hit must be very similar to avoid
    returning results from a neighbouring but distinct topic.

    A sparse cluster requires a looser threshold: its documents are
    naturally more spread, so paraphrase queries may score lower cosine
    similarity while still being semantically equivalent.

    Formula: theta_k = clip(mean_k + alpha * std_k, lower, upper)

    Where mean_k and std_k are derived from pairwise cosine similarities
    among documents whose dominant cluster (argmax of membership) is k.

    Embeddings are assumed to be L2-normalized (MiniLM encodes with
    normalize_embeddings=True), so dot product = cosine similarity.
    """
    n_clusters = membership_matrix.shape[1]
    dominant_clusters = np.argmax(membership_matrix, axis=1)
    thresholds: Dict[int, float] = {}

    logger.info(
        "Computing adaptive thresholds | n_clusters=%d | alpha=%.2f | sample_cap=%d",
        n_clusters, alpha, _MAX_SAMPLE_PER_CLUSTER
    )

    for k in range(n_clusters):
        member_indices = np.where(dominant_clusters == k)[0]
        n_members = len(member_indices)

        if n_members < 2:
            # Cannot compute pairwise distribution — use conservative default
            thresholds[k] = 0.75
            logger.warning(
                "Cluster %d has %d member(s) — insufficient for pairwise computation. "
                "Assigned default threshold=0.75",
                k, n_members
            )
            continue

        # Sample if cluster exceeds cap to keep computation O(sample^2) not O(N^2)
        if n_members > _MAX_SAMPLE_PER_CLUSTER:
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(n_members, _MAX_SAMPLE_PER_CLUSTER, replace=False)
            member_indices = member_indices[sample_idx]

        cluster_embeddings = embeddings[member_indices]

        # Pairwise cosine similarity matrix via matrix multiplication
        # Valid because embeddings are L2-normalized: dot(u,v) = cos(u,v)
        sim_matrix = cluster_embeddings @ cluster_embeddings.T

        # Extract upper triangle only — avoids duplicate pairs and self-similarity (1.0)
        upper_tri_idx = np.triu_indices(len(cluster_embeddings), k=1)
        pairwise_sims = sim_matrix[upper_tri_idx]

        mean_sim = float(np.mean(pairwise_sims))
        std_sim = float(np.std(pairwise_sims))
        raw_threshold = mean_sim + alpha * std_sim
        clamped = float(np.clip(raw_threshold, _LOWER_BOUND, _UPPER_BOUND))

        thresholds[k] = round(clamped, 4)

        logger.info(
            "Cluster %d | n_docs=%d | mean_sim=%.4f | std_sim=%.4f | "
            "raw_theta=%.4f | adaptive_theta=%.4f",
            k, n_members, mean_sim, std_sim, raw_threshold, clamped
        )

    logger.info("Adaptive thresholds computed: %s", thresholds)
    return thresholds


def save_adaptive_thresholds(thresholds: Dict[int, float], path: Path) -> None:
    # JSON requires string keys
    str_keyed = {str(k): v for k, v in thresholds.items()}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(str_keyed, fh, indent=2)
    logger.info("Adaptive thresholds saved | path=%s", path)


def load_adaptive_thresholds(path: Path) -> Dict[int, float]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    # Convert string keys back to int
    thresholds = {int(k): float(v) for k, v in data.items()}
    logger.info("Adaptive thresholds loaded | %s", thresholds)
    return thresholds
