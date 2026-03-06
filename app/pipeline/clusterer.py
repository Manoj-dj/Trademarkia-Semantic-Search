import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from logger.logging_config import get_logger

logger = get_logger(__name__)


def fit_pca(embeddings: np.ndarray, n_components: int) -> Tuple[PCA, np.ndarray]:
    """
    Apply PCA dimensionality reduction before GMM fitting.

    Rationale:
    MiniLM produces 384-dimensional vectors. Fitting a GMM with full covariance
    matrices in 384-D space requires estimating 384x384 covariance matrices per
    component — in practice these become near-singular (ill-conditioned) because
    the number of parameters grows as O(K * D^2) while the number of data points
    is fixed. PCA to 64 dimensions:
    1. Reduces covariance matrix size from 384x384 to 64x64 per component
    2. Concentrates variance in the leading components, improving EM convergence
    3. Reduces clustering wall-clock time significantly
    Typical variance retained at 64 components for MiniLM embeddings: ~85-90%.
    """
    logger.info(
        "Fitting PCA | input_dim=%d | output_dim=%d | n_samples=%d",
        embeddings.shape[1],
        n_components,
        len(embeddings),
    )
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    explained = float(pca.explained_variance_ratio_.sum())
    logger.info("PCA complete | explained_variance=%.4f", explained)
    return pca, reduced.astype(np.float32)


def select_optimal_k(
    reduced_embeddings: np.ndarray,
    k_min: int,
    k_max: int,
) -> Tuple[int, Dict[int, float]]:
    """
    Select GMM cluster count K via Bayesian Information Criterion (BIC) sweep.

    BIC = -2 * log_likelihood + n_parameters * log(n_samples)
    It penalizes model complexity proportionally to log(N), preventing
    the trivially best solution of K=N. The K minimizing BIC represents
    the best balance between data fit and model parsimony.

    We do NOT default to K=20 (the number of original categories).
    Semantic structure in embedding space may differ substantially from
    the original taxonomy. For example, comp.sys.mac.hardware and
    comp.sys.ibm.pc.hardware often collapse into a single hardware cluster,
    while talk.politics.* categories may split along finer ideological lines.
    The BIC sweep lets the data determine K, not our prior assumptions.
    """
    logger.info("BIC sweep | k_range=[%d, %d]", k_min, k_max)
    bic_scores: Dict[int, float] = {}
    best_k = k_min
    best_bic = float("inf")

    for k in range(k_min, k_max + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                max_iter=150,
                random_state=42,
                n_init=3,
            )
            gmm.fit(reduced_embeddings)
            bic = float(gmm.bic(reduced_embeddings))
            bic_scores[k] = bic
            logger.debug("K=%d | BIC=%.2f", k, bic)
            if bic < best_bic:
                best_bic = bic
                best_k = k
        except Exception as exc:
            logger.warning("GMM failed at K=%d | error=%s", k, str(exc))

    logger.info("BIC sweep complete | optimal_k=%d | best_bic=%.2f", best_k, best_bic)
    return best_k, bic_scores


def fit_gmm(reduced_embeddings: np.ndarray, n_components: int) -> GaussianMixture:
    """
    Fit the final GMM with the BIC-selected K.

    Algorithm choice rationale:
    GMM was chosen over Fuzzy C-Means because:
    - GMM provides a principled probabilistic model: output is P(cluster | doc),
      a proper posterior distribution derived from Bayes' theorem via EM.
    - FCM is distance-based (minimizes weighted sum of distances to centroids)
      and assumes spherical clusters — a poor fit for high-dimensional embedding
      space where cluster shapes are irregular.
    - GMM with full covariance handles ellipsoidal, overlapping clusters naturally.
    - scikit-learn's GaussianMixture is well-tested and numerically stable.

    n_init=5 runs EM from 5 different random initializations and keeps the best
    result by log-likelihood, reducing sensitivity to poor initialization.
    """
    logger.info(
        "Fitting final GMM | n_components=%d | n_samples=%d",
        n_components,
        len(reduced_embeddings),
    )
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        max_iter=300,
        random_state=42,
        n_init=5,
    )
    gmm.fit(reduced_embeddings)
    logger.info(
        "GMM fit complete | converged=%s | lower_bound=%.4f",
        gmm.converged_,
        gmm.lower_bound_,
    )
    return gmm


def compute_membership_matrix(
    gmm: GaussianMixture,
    reduced_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Compute the full soft membership matrix.
    Shape: (N_documents, K). Each row sums to 1.0.
    Row i is the posterior distribution P(cluster_j | document_i) for all j.
    """
    logger.info("Computing membership matrix | n_docs=%d", len(reduced_embeddings))
    matrix = gmm.predict_proba(reduced_embeddings).astype(np.float32)
    logger.info("Membership matrix computed | shape=%s", matrix.shape)
    return matrix


def predict_query_membership(
    gmm: GaussianMixture,
    pca: PCA,
    query_embedding: np.ndarray,
) -> np.ndarray:
    """
    Get the fuzzy cluster membership distribution for a single query embedding.
    Projects the query through PCA then applies GMM predict_proba.
    Returns shape (K,).
    """
    reduced = pca.transform(query_embedding)
    probs = gmm.predict_proba(reduced)
    return probs[0].astype(np.float32)


def save_artifacts(
    gmm: GaussianMixture,
    pca: PCA,
    membership_matrix: np.ndarray,
    bic_scores: Dict[int, float],
    gmm_path: Path,
    pca_path: Path,
    membership_path: Path,
    bic_path: Path,
) -> None:
    logger.info("Saving clustering artifacts")
    with open(gmm_path, "wb") as fh:
        pickle.dump(gmm, fh)
    with open(pca_path, "wb") as fh:
        pickle.dump(pca, fh)
    np.save(str(membership_path), membership_matrix)
    with open(bic_path, "wb") as fh:
        pickle.dump(bic_scores, fh)
    logger.info(
        "Artifacts saved | gmm=%s | pca=%s | membership=%s",
        gmm_path,
        pca_path,
        membership_path,
    )


def load_artifacts(
    gmm_path: Path,
    pca_path: Path,
    membership_path: Path,
) -> Tuple[GaussianMixture, PCA, np.ndarray]:
    logger.info("Loading clustering artifacts")
    with open(gmm_path, "rb") as fh:
        gmm = pickle.load(fh)
    with open(pca_path, "rb") as fh:
        pca = pickle.load(fh)
    membership_matrix = np.load(str(membership_path))
    logger.info(
        "Clustering artifacts loaded | gmm_components=%d", gmm.n_components
    )
    return gmm, pca, membership_matrix
