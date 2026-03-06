import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer

from logger.logging_config import get_logger

logger = get_logger(__name__)

# Model selection rationale:
# all-MiniLM-L6-v2 produces 384-dimensional sentence embeddings.
# It runs entirely on CPU with ~33M parameters, making it suitable for
# a lightweight offline pipeline. It consistently ranks among the top
# small models on the MTEB semantic similarity benchmark. Most importantly,
# it runs fully offline — no API keys, no network latency, reproducible
# across environments. Heavier alternatives (BGE, E5, MPNet) offer marginal
# gains in retrieval quality at significant cost in inference speed for
# a corpus this size (~20K documents).
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_BATCH_SIZE = 256


class Embedder:
    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", _MODEL_NAME)
        self._model = SentenceTransformer(_MODEL_NAME)
        logger.info("Embedding model loaded")

    def encode_corpus(self, texts: List[str]) -> np.ndarray:
        """
        Encode all corpus documents into L2-normalized float32 embeddings.

        normalize_embeddings=True applies L2 normalization at encoding time.
        This is required so that FAISS IndexFlatIP (inner product) computes
        cosine similarity correctly: dot(u, v) == cosine_sim(u, v) when both
        vectors are unit-normalized.

        Returns shape (N, 384) float32.
        """
        logger.info(
            "Encoding %d documents | batch_size=%d", len(texts), _BATCH_SIZE
        )
        embeddings = self._model.encode(
            texts,
            batch_size=_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        result = embeddings.astype(np.float32)
        logger.info("Corpus encoding complete | shape=%s", result.shape)
        return result

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string.
        Returns shape (1, 384) float32 for direct FAISS search compatibility.
        """
        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)
