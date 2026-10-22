import numpy as np
import faiss
from legacy.data import HOTELS
from legacy.utils import parseData
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from pathlib import Path
from botocore.exceptions import ClientError
import json
import logging
from functools import lru_cache
from constants import *
from storage.s3 import downloadFromS3, uploadToS3

load_dotenv(override=True)
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class VectorStore:
    index: faiss.Index
    docs: list[dict]


@lru_cache(maxsize=1)
def load_model() -> SentenceTransformer:
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
    )


DIM = 384
FORCE_REINGEST = os.environ.get("FORCE_REINGEST") == "True"


def loadDataToIndex(data: dict, index: faiss.IndexFlatIP):
    # Loads data to FAISS index
    for name, hotel in data.items():
        hotel_info = parseData({name: hotel})

        hotel_vector = embed_text(hotel_info)
        index.add(hotel_vector)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))


def createHotelsDocument(data: dict) -> list[dict]:
    # Create hotel document, same order as the vectors in the FAISS index
    # (doc position == FAISS index id)
    Hotels_doc = [{name: hotel} for name, hotel in data.items()]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_PATH.write_text(json.dumps(Hotels_doc, indent=4), encoding="utf-8")

    return Hotels_doc


def buildIndex(data: dict = HOTELS) -> tuple[faiss.Index, list[dict]]:
    """
    Ingestion pipeline:
      - Parse hotels data, insert to Faiss Index vector store
      - Creates the hotels doc
      - Saves both to ARTIFACT_DIR
    """
    index = createFaissIndex()
    loadDataToIndex(data, index)
    docs = createHotelsDocument(data)
    return index, docs


def loadOrBuildIndex(version: int = 1) -> VectorStore:
    """
    Try: download the index and the hotels doc from S3
    Except (not in S3) or FORCE_REINGEST:
        run the ingestion pipeline and upload both to S3
    """
    if not FORCE_REINGEST:
        try:
            downloadFromS3(version)
            logger.info("Index v%s downloaded from S3", version)
            index = faiss.read_index(str(INDEX_PATH))
            docs = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
            return VectorStore(index, docs)

        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                raise
            logger.warning("Index v%s not found in S3. Running ingestion...", version)

    index, docs = buildIndex()
    uploadToS3(version)
    logger.info("Index v%s uploaded to S3 (%s vectors)", version, index.ntotal)
    return VectorStore(index, docs)


def createFaissIndex(dim=DIM):
    """
    FAISS vector index containing document embeddings.
    """
    index = faiss.IndexFlatIP(dim)
    return index


def embed_text(text: str) -> np.ndarray:
    embedded_text = load_model().encode(text, normalize_embeddings=True)
    return embedded_text.reshape(1, -1)


def similaritySearch(
    question: str, v_store: VectorStore, top_k: int = 3, threshold: float = 0.05
) -> list[int]:
    """
    Performs semantic similarity search using FAISS

    Convert the input question in to an embedding and
    retrieves the indexes of the top-k most semantically
    similar chunks

    Args
        question (str): User's natural language query.

    Returns:
        list[int]: Indices of the most relevant documents,
        (highest first).
    """
    question_embedding = embed_text(question)

    scores, indexes = v_store.index.search(question_embedding, k=top_k)
    logger.debug(
        "search ids=%s scores=%s threshold=%s",
        indexes[0].tolist(), [round(float(s), 3) for s in scores[0]], threshold,
    )
    # kb_index: index referring the Knowledge base
    # i: index of the 'indexes' subset
    indexes = [
        kb_index
        for i, kb_index in enumerate(indexes.tolist()[0])
        if scores[0][i] >= threshold and kb_index != -1
    ]
    logger.debug("search kept ids=%s", indexes)

    return [v_store.docs[i] for i in indexes], indexes
