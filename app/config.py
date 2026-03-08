from pathlib import Path
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    cache_sim_threshold: float = 0.70
    cache_multi_bucket_threshold: float = 0.2
    top_k_results: int = 5
    pca_components: int = 64
    gmm_k_min: int = 5
    gmm_k_max: int = 40
    log_level: str = "INFO"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

ARTIFACTS_DIR = BASE_DIR / "data" / "artifacts"
METADATA_DIR = BASE_DIR / "data" / "metadata"

FAISS_INDEX_PATH = ARTIFACTS_DIR / "faiss_index.bin"
DOC_EMBEDDINGS_PATH = ARTIFACTS_DIR / "doc_embeddings.npy"
CLUSTER_MODEL_PATH = ARTIFACTS_DIR / "cluster_model.pkl"
PCA_MODEL_PATH = ARTIFACTS_DIR / "pca_model.pkl"
CLUSTER_MEMBERSHIP_PATH = ARTIFACTS_DIR / "cluster_membership.npy"
METADATA_MAP_PATH = METADATA_DIR / "metadata_map.json"
BIC_SCORES_PATH = ARTIFACTS_DIR / "bic_scores.pkl"

# NEW: adaptive per-cluster threshold artifact
ADAPTIVE_THRESHOLDS_PATH = ARTIFACTS_DIR / "adaptive_thresholds.json"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
