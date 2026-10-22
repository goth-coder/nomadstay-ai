"""
Embedding models for the v2 chunk index. All are 384-d, so the FAISS config
does not change between them; the manifest records which one built an index.
"""

import logging
import time
from dataclasses import dataclass
from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DIM = 384
MAX_SEQ_LENGTH = 256  # tokens; ~600 chars of text fit, longer is truncated


@dataclass(frozen=True)
class EmbeddingModel:
    key: str
    name: str
    dim: int = DIM
    query_prefix: str = ""
    passage_prefix: str = ""


EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    m.key: m
    for m in [
        # v1 model, English only
        EmbeddingModel("minilm", "sentence-transformers/all-MiniLM-L6-v2"),
        EmbeddingModel("multilingual-minilm",
                       "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
        EmbeddingModel("e5-small", "intfloat/multilingual-e5-small",
                       query_prefix="query: ", passage_prefix="passage: "),
    ]
}


def getModel(key: str) -> EmbeddingModel:
    if key not in EMBEDDING_MODELS:
        raise ValueError(f"unknown embedding model {key!r}, use one of {sorted(EMBEDDING_MODELS)}")
    return EMBEDDING_MODELS[key]


@cache
def loadModel(key: str) -> SentenceTransformer:
    started = time.time()
    model = SentenceTransformer(getModel(key).name, device="cpu")
    model.max_seq_length = MAX_SEQ_LENGTH
    logger.info("loaded %s in %.1fs", key, time.time() - started)
    return model


def embedTexts(texts: list[str], key: str, kind: str = "passage",
               batch_size: int = 64) -> np.ndarray:
    """Texts -> float32 (n, dim), L2-normalized, with the model's query/passage prefix."""
    if kind not in ("passage", "query"):
        raise ValueError(f"kind must be 'passage' or 'query', got {kind!r}")
    spec = getModel(key)
    prefix = spec.query_prefix if kind == "query" else spec.passage_prefix
    if not texts:
        return np.zeros((0, spec.dim), dtype=np.float32)
    vectors = loadModel(key).encode(
        [prefix + t for t in texts], batch_size=batch_size,
        normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
    )
    return np.ascontiguousarray(vectors, dtype=np.float32)
