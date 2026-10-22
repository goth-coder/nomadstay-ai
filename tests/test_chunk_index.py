import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from constants import CITY_CONFIG, refinedDir
from domain.schemas import Chunk
from ingest import osm
from ingest.build import buildArtifacts
from ingest.insideairbnb import ingest
from retrieval import embeddings
from retrieval.chunk_index import (
    ARTIFACT_FILES,
    CHUNKS_FILE,
    INDEX_FILE,
    MANIFEST,
    loadChunkIndex,
    search,
    sha256File,
)
from storage import s3

FIXTURES = Path(__file__).parent / "fixtures"
MINI = FIXTURES / "mini_porto"
EXPECTED = json.loads((MINI / "expected.json").read_text(encoding="utf-8"))
SNAPSHOT = EXPECTED["snapshot_date"]
MODEL = "multilingual-minilm"  # default chosen by the T3 benchmark
WIFI_DROPS = "32235516"  # Bolhão Central Studio, complains about wifi
FRENCH = "40001013"


@pytest.fixture(scope="module")
def store(built):
    return loadChunkIndex(built[0], MODEL)


@pytest.fixture
def copy(built, tmp_path):
    # A throwaway copy of the artifact set that a test may corrupt
    target = tmp_path / "set"
    shutil.copytree(built[0], target)
    return target


# ---------- build ----------

def test_build_writes_the_artifact_set(built):
    out, manifest = built
    assert sorted(p.name for p in out.iterdir()) == sorted(ARTIFACT_FILES)
    assert refinedDir("porto", 2).parts[-3:] == ("refined", "v2", "porto")

    chunks = pd.read_parquet(out / CHUNKS_FILE)
    assert list(chunks.columns) == list(Chunk.model_fields)
    assert chunks["chunk_id"].tolist() == list(range(len(chunks)))
    assert manifest["faiss_ntotal"] == len(chunks) == manifest["counts"]["chunks"]
    assert manifest["counts"]["by_source"] == chunks["source"].value_counts().to_dict()
    assert set(manifest["counts"]["by_field"]) == {"description", "neighborhood_overview", "comments"}


def test_manifest_has_model_sources_and_file_hashes(built):
    out, manifest = built
    saved = json.loads((out / MANIFEST).read_text(encoding="utf-8"))
    assert saved == manifest
    assert saved["embedding"] == {
        "key": MODEL, "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dim": 384, "query_prefix": "", "passage_prefix": "", "normalized": True,
        "max_seq_length": 256,
    }
    assert saved["version"] == 2 and saved["city"] == "porto"
    assert saved["snapshot_date"] == CITY_CONFIG["porto"]["snapshot"]
    assert saved["sources"]["inside_airbnb"]["license"] == "CC BY 4.0"
    assert saved["sources"]["openstreetmap"]["attribution"] == osm.ATTRIBUTION
    assert saved["chunking"]["max_chars"] == 600
    assert "git_commit" in saved and saved["built_at"]
    for name, info in saved["files"].items():
        assert info["bytes"] == (out / name).stat().st_size
        assert info["sha256"] == sha256File(out / name)


def test_build_refuses_listings_without_nearest_gym(tmp_path):
    # Re-running ingest.insideairbnb wipes the OSM columns
    ingest(MINI / "listings.csv", MINI / "reviews.csv", tmp_path, "porto", SNAPSHOT)
    with pytest.raises(SystemExit, match="ingest.osm"):
        buildArtifacts(tmp_path, tmp_path / "out", "porto", MODEL, 2)
    assert not (tmp_path / "out").exists()


def test_limit_builds_a_partial_index(processed, tmp_path):
    manifest = buildArtifacts(processed, tmp_path, "porto", MODEL, 2, limit=3)
    chunks = pd.read_parquet(tmp_path / CHUNKS_FILE)
    assert chunks["listing_id"].nunique() == 3
    assert manifest["limit_listings"] == 3 and manifest["counts"]["listings"] == 3


# ---------- embeddings ----------

def test_embedTexts_is_normalized_float32_384():
    vectors = embeddings.embedTexts(["first text", "second one", "third"], MODEL)
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


def test_embedTexts_empty_and_bad_kind():
    assert embeddings.embedTexts([], MODEL).shape == (0, 384)
    with pytest.raises(ValueError, match="kind"):
        embeddings.embedTexts(["x"], MODEL, kind="document")
    with pytest.raises(ValueError, match="unknown embedding model"):
        embeddings.embedTexts(["x"], "gpt")


def test_e5_prefixes_are_applied(monkeypatch):
    seen = []

    class FakeModel:
        def encode(self, texts, **kwargs):
            seen.append(list(texts))
            return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(embeddings, "loadModel", lambda key: FakeModel())
    embeddings.embedTexts(["wifi"], "e5-small", kind="query")
    embeddings.embedTexts(["wifi"], "e5-small", kind="passage")
    embeddings.embedTexts(["wifi"], "minilm", kind="query")
    assert seen == [["query: wifi"], ["passage: wifi"], ["wifi"]]


# ---------- loader ----------

def test_loader_returns_aligned_structures(store, built):
    assert store.index.ntotal == len(store.chunks) == built[1]["faiss_ntotal"]
    assert store.manifest["embedding"]["key"] == MODEL
    for listing_id, positions in store.positions_by_listing.items():
        assert (store.chunks.iloc[positions]["listing_id"] == listing_id).all()
    assert sum(len(p) for p in store.positions_by_listing.values()) == len(store.chunks)


def test_loader_refuses_other_runtime_model(built):
    with pytest.raises(ValueError, match="embedding key"):
        loadChunkIndex(built[0], "minilm")


def test_loader_refuses_dim_or_name_mismatch(copy):
    manifest = json.loads((copy / MANIFEST).read_text(encoding="utf-8"))
    manifest["embedding"]["dim"] = 768
    (copy / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="dim"):
        loadChunkIndex(copy, MODEL)

    manifest["embedding"]["dim"] = 384
    manifest["embedding"]["name"] = "another/model"
    (copy / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="name"):
        loadChunkIndex(copy, MODEL)


def test_loader_refuses_hash_mismatch(copy):
    with open(copy / CHUNKS_FILE, "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="hash"):
        loadChunkIndex(copy, MODEL)


def test_loader_refuses_missing_file(copy):
    (copy / INDEX_FILE).unlink()
    with pytest.raises(ValueError, match="missing"):
        loadChunkIndex(copy, MODEL)


def test_loader_refuses_ntotal_mismatch(copy):
    # Fewer chunk rows than vectors, with a consistent manifest hash: only ntotal catches it
    chunks = pd.read_parquet(copy / CHUNKS_FILE).head(-1)
    chunks.to_parquet(copy / CHUNKS_FILE, index=False)
    manifest = json.loads((copy / MANIFEST).read_text(encoding="utf-8"))
    manifest["files"][CHUNKS_FILE]["sha256"] = sha256File(copy / CHUNKS_FILE)
    (copy / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="FAISS has"):
        loadChunkIndex(copy, MODEL)


# ---------- search ----------

def test_search_returns_chunk_fields_and_scores_best_first(store):
    results = search(store, "the wifi kept disconnecting during calls", k=5)
    assert len(results) == 5
    assert set(results[0]) == set(Chunk.model_fields) | {"score"}
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(r["chunk_id"], int) for r in results)
    # chunk_id is the position: the stored text is what the result says
    assert all(store.chunks.iloc[r["chunk_id"]]["text"] == r["text"] for r in results)


def test_search_finds_non_english_review_with_multilingual_model(store):
    results = search(store, "fast internet good for video calls", k=5)
    assert any(r["text"].startswith("Internet excelente") for r in results)


def test_search_with_allowed_listings_only_returns_those(store):
    allowed = {WIFI_DROPS, FRENCH}
    results = search(store, "wifi problems", k=20, allowed_listing_ids=allowed)
    assert results and {r["listing_id"] for r in results} <= allowed
    wanted = sum(len(store.positions_by_listing[i]) for i in allowed)
    assert len(results) == min(20, wanted)


def test_search_filter_changes_the_winner(store):
    # The unfiltered best match for wifi is elsewhere; the filter forces another listing
    best = search(store, "the wifi dropped constantly", k=1)[0]
    other = (set(store.positions_by_listing) - {best["listing_id"]})
    forced = search(store, "the wifi dropped constantly", k=1, allowed_listing_ids=other)
    assert forced[0]["listing_id"] != best["listing_id"]


def test_search_empty_or_unknown_allowed_listings_return_nothing(store):
    assert search(store, "wifi", allowed_listing_ids=set()) == []
    assert search(store, "wifi", allowed_listing_ids={"not-a-listing"}) == []


def test_search_listing_chunks_have_no_review_fields(store):
    results = search(store, "bright apartment", k=40, allowed_listing_ids={WIFI_DROPS})
    listing = [r for r in results if r["source"] == "listing"]
    assert listing and all(r["review_id"] is None and r["date"] is None for r in listing)
    review = [r for r in results if r["source"] == "review"]
    assert review and all(r["review_id"] and r["date"] for r in review)


# ---------- S3 artifact functions (fake client) ----------

class FakeS3:
    """Stands in for boto3's S3 client: a dict of key -> bytes, no network."""

    def __init__(self, objects=None):
        self.objects = dict(objects or {})

    def upload_file(self, path, bucket, key):
        self.objects[(bucket, key)] = Path(path).read_bytes()

    def download_file(self, bucket, key, path):
        if (bucket, key) not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        Path(path).write_bytes(self.objects[(bucket, key)])


@pytest.fixture
def fakeS3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(s3, "session", SimpleNamespace(client=lambda name: fake))
    return fake


def test_artifact_keys():
    assert s3.artifactKey(2, "porto", "chunks.index") == "rag/v2/porto/chunks.index"


def test_upload_and_download_artifacts_roundtrip(built, fakeS3, tmp_path):
    keys = s3.uploadArtifacts(built[0], 2, "porto", ARTIFACT_FILES)
    assert keys == [f"rag/v2/porto/{n}" for n in ARTIFACT_FILES]
    assert set(fakeS3.objects) == {(s3.BUCKET, k) for k in keys}

    paths = s3.downloadArtifacts(2, "porto", tmp_path / "downloaded", ARTIFACT_FILES)
    assert [p.name for p in paths] == ARTIFACT_FILES
    for name in ARTIFACT_FILES:
        assert (tmp_path / "downloaded" / name).read_bytes() == (built[0] / name).read_bytes()
    loadChunkIndex(tmp_path / "downloaded", MODEL)  # the downloaded set is loadable


def test_download_artifacts_missing_version_raises_404(fakeS3, tmp_path):
    with pytest.raises(ClientError) as error:
        s3.downloadArtifacts(9, "porto", tmp_path, ["chunks.index"])
    assert error.value.response["Error"]["Code"] == "404"
