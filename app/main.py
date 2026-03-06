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
from app.cache.semantic_cache import SemanticCache
from logger.logging_config import get_logger

logger = get_logger(__name__)


def _all_artifacts_present() -> bool:
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
    Full one-time build pipeline. Invoked automatically on first server start
    when no artifacts are found. Subsequent starts skip this and load from disk.

    Stages:
    1. Load and clean 20 Newsgroups corpus
    2. Encode documents with MiniLM (L2-normalized, 384-dim)
    3. Build FAISS IndexFlatIP for semantic retrieval
    4. Reduce embeddings via PCA (384 -> pca_components) for GMM stability
    5. BIC sweep over K=[gmm_k_min, gmm_k_max] to select optimal cluster count
    6. Fit final GMM and compute per-document membership matrix
    7. Build and persist metadata map (FAISS id -> doc text + cluster distribution)

    Expected wall-clock time on CPU: 15-40 minutes depending on hardware.
    All artifacts are persisted to data/artifacts/ and data/metadata/.
    """
    logger.info("Build pipeline started (first run — artifacts not found)")

    # Stage 1: Preprocessing
    documents = load_and_clean()
    texts = [doc["text"] for doc in documents]

    # Stage 2: Embedding
    embedder = Embedder()
    embeddings = embedder.encode_corpus(texts)
    np.save(str(DOC_EMBEDDINGS_PATH), embeddings)
    logger.info("Embeddings persisted | path=%s | shape=%s", DOC_EMBEDDINGS_PATH, embeddings.shape)

    # Stage 3: FAISS index
    faiss_index = build_faiss_index(embeddings)
    save_faiss_index(faiss_index, FAISS_INDEX_PATH)

    # Stage 4: PCA dimensionality reduction
    pca, reduced_embeddings = fit_pca(embeddings, n_components=settings.pca_components)

    # Stage 5: BIC-based cluster count selection
    optimal_k, bic_scores = select_optimal_k(
        reduced_embeddings,
        k_min=settings.gmm_k_min,
        k_max=settings.gmm_k_max,
    )

    # Stage 6: Final GMM + membership matrix
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

    # Stage 7: Metadata map
    metadata_map = build_metadata_map(documents, membership_matrix)
    save_metadata_map(metadata_map, METADATA_MAP_PATH)

    logger.info("Build pipeline complete | all artifacts persisted")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Startup:
    - Check for pre-built artifacts; run full build pipeline if absent.
    - Load FAISS index, GMM, PCA, metadata map, and embedding model into app.state.
    - Initialize the SemanticCache singleton with configured threshold values.
    All objects are loaded once at startup and reused across all requests.
    Using app.state (not module-level globals) ensures clean testability
    and proper resource scoping per application instance.

    Shutdown:
    - Python garbage collector handles model cleanup.
    - Log clean exit.

    Rationale for lifespan over @app.on_event:
    @app.on_event('startup') is deprecated in FastAPI >= 0.93.
    The lifespan pattern is the current recommended approach per FastAPI docs.
    """
    logger.info("FastAPI startup sequence initiated")

    if not _all_artifacts_present():
        logger.info(
            "Artifacts not found in %s — running full build pipeline. "
            "This will take 15-40 minutes on first run.",
            ARTIFACTS_DIR,
        )
        _run_build_pipeline()
    else:
        logger.info("Pre-built artifacts found — skipping build pipeline")

    logger.info("Loading artifacts into app.state")

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

    app.state.cache = SemanticCache(
        sim_threshold=settings.cache_sim_threshold,
        multi_bucket_threshold=settings.cache_multi_bucket_threshold,
    )

    logger.info("All artifacts loaded. Server is ready to serve requests.")
    yield
    logger.info("FastAPI shutdown complete")


def create_app() -> FastAPI:
    application = FastAPI(
        title="Trademarkia Semantic Search API",
        description=(
            "Lightweight semantic search over the 20 Newsgroups corpus. "
            "Implements fuzzy GMM clustering, cluster-partitioned semantic "
            "cache (built from first principles), and FAISS-based vector retrieval."
        ),
        version="1.0.0",
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
