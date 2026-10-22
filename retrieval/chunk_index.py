"""
Chunk index of the v2 artifact set: FAISS vectors + chunks.parquet + manifest.

Row `i` of chunks.parquet is FAISS position `i`. The loader refuses an artifact
set whose embedding model, dimension, size or file hashes do not match.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from retrieval.embeddings import embedTexts, getModel

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
INDEX_FILE = "chunks.index"
CHUNKS_FILE = "chunks.parquet"
LISTINGS_FILE = "listings.parquet"
POIS_FILE = "pois.parquet"
ARTIFACT_FILES = [INDEX_FILE, CHUNKS_FILE, LISTINGS_FILE, POIS_FILE, MANIFEST]


@dataclass
class ChunkIndex:
    index: faiss.Index
    chunks: pd.DataFrame
    manifest: dict
    positions_by_listing: dict[str, np.ndarray]  # listing_id -> FAISS positions of its chunks


def sha256File(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def checkManifest(directory: Path, manifest: dict, model_key: str) -> None:
    """Raise ValueError if the manifest does not match the runtime model or the files on disk."""
    runtime = getModel(model_key)
    built = manifest.get("embedding", {})
    for field, want in [("key", runtime.key), ("name", runtime.name), ("dim", runtime.dim)]:
        if built.get(field) != want:
            raise ValueError(
                f"index was built with embedding {field}={built.get(field)!r}, "
                f"runtime model has {want!r}")

    for name, info in manifest.get("files", {}).items():
        path = directory / name
        if not path.exists():
            raise ValueError(f"{name} is listed in the manifest but missing")
        if sha256File(path) != info["sha256"]:
            raise ValueError(f"{name} does not match the manifest hash")


def loadChunkIndex(directory: str | Path, model_key: str) -> ChunkIndex:
    directory = Path(directory)
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    checkManifest(directory, manifest, model_key)

    index = faiss.read_index(str(directory / INDEX_FILE))
    chunks = pd.read_parquet(directory / CHUNKS_FILE)
    if index.d != getModel(model_key).dim:
        raise ValueError(f"FAISS dimension {index.d} != model dimension {getModel(model_key).dim}")
    if index.ntotal != len(chunks) or manifest.get("faiss_ntotal") != index.ntotal:
        raise ValueError(
            f"FAISS has {index.ntotal} vectors, chunks.parquet {len(chunks)} rows, "
            f"manifest {manifest.get('faiss_ntotal')}")
    if not (chunks["chunk_id"].to_numpy() == np.arange(len(chunks))).all():
        raise ValueError("chunk_id is not the row position")

    positions = {listing_id: np.asarray(rows, dtype=np.int64)
                 for listing_id, rows in chunks.groupby("listing_id").indices.items()}
    logger.info("chunk index loaded: %s chunks, %s listings", index.ntotal, len(positions))
    return ChunkIndex(index, chunks, manifest, positions)


def valueOrNone(value):
    if pd.isna(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def search(store: ChunkIndex, query: str, k: int = 10,
           allowed_listing_ids: set[str] | None = None) -> list[dict]:
    """
    The k chunks closest to the query (chunk fields + score, best first).
    allowed_listing_ids limits the search to those listings' chunks; an empty
    set returns nothing. The filter runs inside FAISS, so k results come back
    whenever that many allowed chunks exist.
    """
    params = None
    if allowed_listing_ids is not None:
        parts = [store.positions_by_listing[i] for i in allowed_listing_ids
                 if i in store.positions_by_listing]
        if not parts:
            return []
        selector = faiss.IDSelectorBatch(np.concatenate(parts))
        params = faiss.SearchParameters(sel=selector)

    vector = embedTexts([query], store.manifest["embedding"]["key"], kind="query")
    scores, positions = store.index.search(vector, k, params=params)

    results = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0:  # fewer than k candidates
            continue
        row = store.chunks.iloc[int(position)]
        results.append({**{name: valueOrNone(row[name]) for name in store.chunks.columns},
                        "score": float(score)})
    return results
