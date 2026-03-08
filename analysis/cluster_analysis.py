"""
Cluster analysis script.

Generates:
1. BIC curve plot — justifies cluster count selection
2. UMAP 2D scatter plot — validates semantic cluster structure visually
3. Boundary document CSV — surfaces semantically ambiguous documents
4. Cluster top-documents CSV — shows what lives in each cluster

Run from project root after the server has started at least once
(so artifacts are built):
    python -m analysis.cluster_analysis
"""

import sys
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (
    CLUSTER_MEMBERSHIP_PATH,
    DOC_EMBEDDINGS_PATH,
    METADATA_MAP_PATH,
    BIC_SCORES_PATH,
)
from logger.logging_config import get_logger

logger = get_logger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_bic_curve() -> int:
    """
    Plot BIC scores vs K.
    The vertical line marks the BIC-optimal K selected during the build pipeline.
    """
    logger.info("Generating BIC curve plot")

    with open(BIC_SCORES_PATH, "rb") as fh:
        bic_scores: dict = pickle.load(fh)

    ks = sorted(bic_scores.keys())
    bics = [bic_scores[k] for k in ks]
    optimal_k = ks[int(np.argmin(bics))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, bics, marker="o", linewidth=1.5, color="steelblue", markersize=4)
    ax.axvline(x=optimal_k, color="crimson", linestyle="--", label=f"Optimal K={optimal_k}")
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("BIC Score")
    ax.set_title("GMM Cluster Selection via Bayesian Information Criterion (BIC)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "bic_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("BIC plot saved | path=%s | optimal_k=%d", plot_path, optimal_k)

    df = pd.DataFrame({"k": ks, "bic": bics})
    df.to_csv(RESULTS_DIR / "bic_scores.csv", index=False)
    logger.info("BIC scores CSV saved")

    return optimal_k


def plot_umap_clusters() -> None:
    """
    Generate a 2D UMAP scatter plot colored by dominant cluster.
    Subsampled to 5000 documents for speed; UMAP on full 20K is slow.
    """
    try:
        import umap as umap_module
    except ImportError:
        logger.warning("umap-learn not installed. Skipping UMAP visualization.")
        return

    logger.info("Generating UMAP cluster visualization")

    embeddings = np.load(str(DOC_EMBEDDINGS_PATH))
    membership_matrix = np.load(str(CLUSTER_MEMBERSHIP_PATH))
    dominant_clusters = np.argmax(membership_matrix, axis=1)
    n_clusters = membership_matrix.shape[1]

    n_sample = min(5000, len(embeddings))
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(embeddings), n_sample, replace=False)
    sample_emb = embeddings[sample_idx]
    sample_clusters = dominant_clusters[sample_idx]

    logger.info("Running UMAP on %d sampled documents", n_sample)
    reducer = umap_module.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    reduced_2d = reducer.fit_transform(sample_emb)

    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
    fig, ax = plt.subplots(figsize=(14, 10))

    for cluster_id in range(n_clusters):
        mask = sample_clusters == cluster_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            reduced_2d[mask, 0],
            reduced_2d[mask, 1],
            color=colors[cluster_id],
            alpha=0.4,
            s=5,
            label=f"Cluster {cluster_id}",
        )

    ax.set_title("UMAP Projection — Colored by Dominant GMM Cluster")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=6, ncol=2, markerscale=3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "umap_clusters.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info("UMAP plot saved | path=%s", plot_path)


def analyze_boundary_documents() -> None:
    """
    Surface semantically ambiguous boundary documents.

    A boundary document is one where the top-2 cluster membership scores
    differ by less than 0.10. These documents belong to two clusters
    in nearly equal measure — the most compelling evidence that hard
    clustering is insufficient for this corpus.
    """
    logger.info("Analyzing boundary documents")

    membership_matrix = np.load(str(CLUSTER_MEMBERSHIP_PATH))
    with open(METADATA_MAP_PATH, "r", encoding="utf-8") as fh:
        metadata_map = json.load(fh)

    records = []
    for idx in range(len(membership_matrix)):
        probs = membership_matrix[idx]
        sorted_desc = np.sort(probs)[::-1]
        gap = float(sorted_desc[0] - sorted_desc[1])

        if gap < 0.10:
            top_idx = np.argsort(probs)[::-1]
            meta = metadata_map.get(str(idx), {})
            records.append(
                {
                    "doc_id": idx,
                    "text_snippet": meta.get("text", "")[:300],
                    "category": meta.get("category", ""),
                    "cluster_1": int(top_idx[0]),
                    "prob_1": round(float(probs[top_idx[0]]), 4),
                    "cluster_2": int(top_idx[1]),
                    "prob_2": round(float(probs[top_idx[1]]), 4),
                    "gap": round(gap, 4),
                }
            )

    df = pd.DataFrame(records).sort_values("gap").head(100)
    out_path = RESULTS_DIR / "boundary_documents.csv"
    df.to_csv(out_path, index=False)
    logger.info("Boundary docs | found=%d | saved=%s", len(records), out_path)


def cluster_top_documents() -> None:
    """
    For each cluster, extract the 5 documents with the highest membership score.
    These are the most representative members of each cluster.
    """
    logger.info("Extracting top documents per cluster")

    membership_matrix = np.load(str(CLUSTER_MEMBERSHIP_PATH))
    with open(METADATA_MAP_PATH, "r", encoding="utf-8") as fh:
        metadata_map = json.load(fh)

    n_clusters = membership_matrix.shape[1]
    records = []

    for cluster_id in range(n_clusters):
        scores = membership_matrix[:, cluster_id]
        top_indices = np.argsort(scores)[::-1][:5]
        for rank, doc_idx in enumerate(top_indices):
            meta = metadata_map.get(str(doc_idx), {})
            records.append(
                {
                    "cluster_id": cluster_id,
                    "rank": rank + 1,
                    "membership_score": round(float(scores[doc_idx]), 4),
                    "category": meta.get("category", ""),
                    "text_snippet": meta.get("text", "")[:300],
                }
            )

    df = pd.DataFrame(records)
    out_path = RESULTS_DIR / "cluster_top_docs.csv"
    df.to_csv(out_path, index=False)
    logger.info("Cluster top documents saved | path=%s", out_path)
    

def plot_cluster_threshold_distribution() -> None:
    """
    For each cluster, plot the intra-cluster pairwise cosine similarity
    histogram with a vertical line at the adaptive threshold.

    This is the visual evidence justifying why each cluster gets a different
    threshold — the sceptical reviewer can see directly that dense clusters
    have a higher mean similarity distribution than sparse ones.
    """
    logger.info("Generating adaptive threshold distribution plots")

    from app.pipeline.adaptive_threshold import (
        compute_per_cluster_thresholds,
        _MAX_SAMPLE_PER_CLUSTER,
    )

    embeddings = np.load(str(DOC_EMBEDDINGS_PATH))
    membership_matrix = np.load(str(CLUSTER_MEMBERSHIP_PATH))
    dominant_clusters = np.argmax(membership_matrix, axis=1)
    n_clusters = membership_matrix.shape[1]

    # Recompute thresholds to get per-cluster stats for plotting
    thresholds = compute_per_cluster_thresholds(embeddings, membership_matrix)

    records = []
    fig, axes = plt.subplots(
        nrows=int(np.ceil(n_clusters / 2)),
        ncols=2,
        figsize=(14, 4 * int(np.ceil(n_clusters / 2)))
    )
    axes_flat = axes.flatten()

    for k in range(n_clusters):
        member_indices = np.where(dominant_clusters == k)[0]
        n_members = len(member_indices)
        ax = axes_flat[k]

        if n_members < 2:
            ax.set_title(f"Cluster {k} (insufficient data)")
            ax.axis("off")
            records.append({
                "cluster_id": k, "n_docs": n_members,
                "mean_sim": None, "std_sim": None,
                "adaptive_threshold": thresholds.get(k, 0.75)
            })
            continue

        if n_members > _MAX_SAMPLE_PER_CLUSTER:
            rng = np.random.default_rng(seed=42)
            sample_idx = rng.choice(n_members, _MAX_SAMPLE_PER_CLUSTER, replace=False)
            member_indices = member_indices[sample_idx]

        cluster_embs = embeddings[member_indices]
        sim_matrix = cluster_embs @ cluster_embs.T
        upper_tri = np.triu_indices(len(cluster_embs), k=1)
        pairwise_sims = sim_matrix[upper_tri]

        mean_sim = float(np.mean(pairwise_sims))
        std_sim = float(np.std(pairwise_sims))
        adaptive_theta = thresholds.get(k, 0.75)

        ax.hist(pairwise_sims, bins=40, color="steelblue", alpha=0.7, edgecolor="none")
        ax.axvline(x=adaptive_theta, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Adaptive theta={adaptive_theta:.3f}")
        ax.axvline(x=mean_sim, color="orange", linestyle=":", linewidth=1.2,
                   label=f"Mean={mean_sim:.3f}")
        ax.set_title(f"Cluster {k} | n={n_members} | theta={adaptive_theta:.3f}")
        ax.set_xlabel("Pairwise Cosine Similarity")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)

        records.append({
            "cluster_id": k,
            "n_docs": n_members,
            "mean_sim": round(mean_sim, 4),
            "std_sim": round(std_sim, 4),
            "adaptive_threshold": adaptive_theta,
        })

    # Hide unused subplot axes
    for i in range(n_clusters, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.suptitle("Per-Cluster Intra-Similarity Distribution and Adaptive Threshold", y=1.01)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "adaptive_thresholds.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Adaptive threshold plot saved | path=%s", plot_path)

    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "adaptive_thresholds.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Adaptive thresholds CSV saved | path=%s", csv_path)



if __name__ == "__main__":
    logger.info("Cluster analysis started")
    plot_bic_curve()
    plot_umap_clusters()
    analyze_boundary_documents()
    cluster_top_documents()
    plot_cluster_threshold_distribution()   # NEW
    logger.info("Cluster analysis complete | results in %s", RESULTS_DIR)
