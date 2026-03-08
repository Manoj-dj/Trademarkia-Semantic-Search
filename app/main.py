from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    settings,
    ARTIFACTS_DIR,
    FAISS_INDEX_PATH,
    DOC_EMBEDDINGS_PATH,
    CLUSTER_MODEL_PATH,
    PCA_MODEL_PATH,
    CLUSTER_MEMBERSHIP_PATH,
    METADATA_MAP_PATH,
    BIC_SCORES_PATH,
    ADAPTIVE_THRESHOLDS_PATH,
)
from app.api.routes import router
from app.pipeline.preprocessor import load_and_clean
from app.pipeline.embedder import Embedder
from app.pipeline.indexer import (
    build_faiss_index,
    save_faiss_index,
    load_faiss_index,
    build_metadata_map,
    save_metadata_map,
    load_metadata_map,
)
from app.pipeline.clusterer import (
    fit_pca,
    select_optimal_k,
    fit_gmm,
    compute_membership_matrix,
    save_artifacts,
    load_artifacts,
)
from app.pipeline.adaptive_threshold import (
    compute_per_cluster_thresholds,
    save_adaptive_thresholds,
    load_adaptive_thresholds,
)
from app.cache.semantic_cache import SemanticCache
from logger.logging_config import get_logger

logger = get_logger(__name__)


def _all_artifacts_present() -> bool:
    """
    Check whether all core artifacts exist.
    Adaptive thresholds are handled separately in lifespan — they can be
    computed on-the-fly from existing artifacts without a full rebuild.
    """
    required = [
        FAISS_INDEX_PATH,
        DOC_EMBEDDINGS_PATH,
        CLUSTER_MODEL_PATH,
        PCA_MODEL_PATH,
        CLUSTER_MEMBERSHIP_PATH,
        METADATA_MAP_PATH,
    ]
    return all(p.exists() for p in required)


def _run_build_pipeline() -> None:
    """
    Full one-time build pipeline. Runs only when core artifacts are absent.

    Stages:
    1. Load and preprocess 20 Newsgroups corpus
    2. Encode with MiniLM (L2-normalized, 384-dim)
    3. Build FAISS IndexFlatIP
    4. PCA reduction (384 -> pca_components dims)
    5. BIC sweep to select optimal K
    6. Fit GMM + compute membership matrix
    7. Build and persist metadata map
    8. Compute and persist adaptive per-cluster thresholds
    """
    logger.info("Build pipeline started")

    documents = load_and_clean()
    texts = [doc["text"] for doc in documents]

    embedder = Embedder()
    embeddings = embedder.encode_corpus(texts)
    np.save(str(DOC_EMBEDDINGS_PATH), embeddings)
    logger.info("Embeddings persisted | shape=%s", embeddings.shape)

    faiss_index = build_faiss_index(embeddings)
    save_faiss_index(faiss_index, FAISS_INDEX_PATH)

    pca, reduced_embeddings = fit_pca(embeddings, n_components=settings.pca_components)

    optimal_k, bic_scores = select_optimal_k(
        reduced_embeddings,
        k_min=settings.gmm_k_min,
        k_max=settings.gmm_k_max,
    )

    gmm = fit_gmm(reduced_embeddings, n_components=optimal_k)
    membership_matrix = compute_membership_matrix(gmm, reduced_embeddings)

    save_artifacts(
        gmm=gmm,
        pca=pca,
        membership_matrix=membership_matrix,
        bic_scores=bic_scores,
        gmm_path=CLUSTER_MODEL_PATH,
        pca_path=PCA_MODEL_PATH,
        membership_path=CLUSTER_MEMBERSHIP_PATH,
        bic_path=BIC_SCORES_PATH,
    )

    metadata_map = build_metadata_map(documents, membership_matrix)
    save_metadata_map(metadata_map, METADATA_MAP_PATH)

    # Stage 8: Adaptive per-cluster thresholds
    # Computed from the L2-normalized corpus embeddings and GMM membership matrix.
    # For each cluster k: theta_k = clip(mean_pairwise_sim_k + alpha * std_k, 0.60, 0.92)
    adaptive_thresholds = compute_per_cluster_thresholds(embeddings, membership_matrix)
    save_adaptive_thresholds(adaptive_thresholds, ADAPTIVE_THRESHOLDS_PATH)

    logger.info("Build pipeline complete | all artifacts persisted")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup sequence:
    1. Run full build pipeline if core artifacts are absent (first run only).
    2. Load all core artifacts into app.state.
    3. Compute adaptive thresholds if missing (incremental — no full rebuild).
    4. Initialize SemanticCache with adaptive thresholds.

    The adaptive threshold file is intentionally separated from the core
    artifact check. If only adaptive_thresholds.json is missing (e.g., after
    upgrading the system), it is recomputed in seconds from existing artifacts
    without re-embedding or re-clustering.
    """
    logger.info("FastAPI startup sequence initiated")

    if not _all_artifacts_present():
        logger.info(
            "Core artifacts not found in %s — running full build pipeline. "
            "This will take 15-40 minutes on first run.",
            ARTIFACTS_DIR,
        )
        _run_build_pipeline()
    else:
        logger.info("Core artifacts found — skipping full build pipeline")

    logger.info("Loading core artifacts into app.state")

    app.state.settings = settings
    app.state.embedder = Embedder()
    app.state.faiss_index = load_faiss_index(FAISS_INDEX_PATH)
    app.state.metadata_map = load_metadata_map(METADATA_MAP_PATH)

    gmm, pca, membership_matrix = load_artifacts(
        gmm_path=CLUSTER_MODEL_PATH,
        pca_path=PCA_MODEL_PATH,
        membership_path=CLUSTER_MEMBERSHIP_PATH,
    )
    app.state.gmm = gmm
    app.state.pca = pca
    app.state.membership_matrix = membership_matrix

    # Load or compute adaptive thresholds independently of core artifact check.
    # This allows the adaptive threshold logic to be added to an existing
    # deployment without triggering a full rebuild.
    if not ADAPTIVE_THRESHOLDS_PATH.exists():
        logger.info(
            "Adaptive thresholds not found — computing from existing artifacts "
            "(embeddings + membership matrix). No full rebuild required."
        )
        embeddings = np.load(str(DOC_EMBEDDINGS_PATH))
        adaptive_thresholds = compute_per_cluster_thresholds(
            embeddings, membership_matrix
        )
        save_adaptive_thresholds(adaptive_thresholds, ADAPTIVE_THRESHOLDS_PATH)
    else:
        logger.info("Adaptive thresholds found — loading from disk")

    adaptive_thresholds = load_adaptive_thresholds(ADAPTIVE_THRESHOLDS_PATH)
    app.state.adaptive_thresholds = adaptive_thresholds

    app.state.cache = SemanticCache(
        sim_threshold=settings.cache_sim_threshold,
        multi_bucket_threshold=settings.cache_multi_bucket_threshold,
        per_cluster_thresholds=adaptive_thresholds,
    )

    logger.info("All artifacts loaded. Server is ready to serve requests.")
    yield
    logger.info("FastAPI shutdown complete")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Trademarkia Semantic Search API",
        description=(
            "Lightweight semantic search over the 20 Newsgroups corpus. "
            "Implements fuzzy GMM clustering, cluster-partitioned semantic cache "
            "with adaptive per-cluster thresholds, and FAISS vector retrieval."
        ),
        version="1.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router, tags=["Semantic Search"])
    return application


app = create_app()
