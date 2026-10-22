"""
Build the v2 artifact set: chunks, embeddings, FAISS index and manifest.

    python -m ingest.build --city porto --model <key> --version 2 [--limit N]
    python -m ingest.build --benchmark

Reads data/processed/<city>/<snapshot>/ (run ingest.insideairbnb, then
ingest.osm) and writes data/refined/v<version>/<city>/. Offline only; it never
uploads (storage.s3.uploadArtifacts does that, after asking).
"""

import argparse
import json
import logging
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from constants import CITY_CONFIG, processedDir, refinedDir
from ingest.chunks import MAX_CHARS, MIN_CHARS, buildChunks
from ingest.insideairbnb import (
    normalizeListings,
    normalizeReviews,
    readListings,
    readListingsCsv,
)
from retrieval.chunk_index import (
    CHUNKS_FILE,
    INDEX_FILE,
    LISTINGS_FILE,
    MANIFEST,
    POIS_FILE,
    sha256File,
)
from retrieval.embeddings import (
    DIM,
    EMBEDDING_MODELS,
    MAX_SEQ_LENGTH,
    embedTexts,
    getModel,
    loadModel,
)

logger = logging.getLogger(__name__)

EMBED_SLICE = 4096  # texts per progress step
FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "mini_porto"


def gitCommit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10, cwd=Path(__file__).parent, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout.strip() or None) if out.returncode == 0 else None


def embedChunks(texts: list[str], key: str, batch_size: int = 32) -> np.ndarray:
    # Embeds in slices so a long build logs its progress
    vectors = np.empty((len(texts), getModel(key).dim), dtype=np.float32)
    started = time.time()
    for start in range(0, len(texts), EMBED_SLICE):
        end = min(start + EMBED_SLICE, len(texts))
        vectors[start:end] = embedTexts(texts[start:end], key, "passage", batch_size)
        rate = end / (time.time() - started)
        logger.info("embedded %s/%s chunks (%.0f/s, ~%.0f min left)",
                    end, len(texts), rate, (len(texts) - end) / rate / 60)
    return vectors


def checkProcessed(listings: pd.DataFrame) -> None:
    # insideairbnb re-runs wipe nearest_gym_m, and the gym aspect depends on it
    if not listings["nearest_gym_m"].notna().any():
        raise SystemExit("listings.parquet has no nearest_gym_m values: "
                         "run `python -m ingest.osm` first")


def manifestSources(report: dict) -> dict:
    ia, osm = report.get("insideairbnb", {}), report.get("osm", {})
    return {
        "inside_airbnb": {
            "url": ia.get("source"), "license": ia.get("license"),
            "snapshot_date": ia.get("snapshot_date"), "raw_files": ia.get("raw_files"),
            "attribution": "Inside Airbnb (insideairbnb.com), CC BY 4.0",
        },
        "openstreetmap": {
            "license": "ODbL", "attribution": osm.get("attribution"),
            "fetched_date": osm.get("fetched_date"),
            "osm_data_timestamp": osm.get("osm_data_timestamp"),
        },
    }


def buildArtifacts(processed: Path, out_dir: Path, city: str, model_key: str, version: int,
                   limit: int | None = None, max_chars: int = MAX_CHARS) -> dict:
    """Processed parquet files -> artifact set in out_dir. Returns the manifest."""
    started = time.time()
    listings = readListings(processed / LISTINGS_FILE)
    checkProcessed(listings)
    reviews = pd.read_parquet(processed / "reviews.parquet")
    if limit:
        listings = listings.head(limit)

    chunks = buildChunks(listings, reviews, max_chars)
    if chunks.empty:
        raise SystemExit("no chunks to index")
    logger.info("%s chunks (%s listings)", len(chunks), len(listings))

    vectors = embedChunks(chunks["text"].tolist(), model_key)
    index = faiss.IndexFlatIP(DIM)
    index.add(vectors)
    assert index.ntotal == len(chunks)  # position <-> row invariant

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / INDEX_FILE))
    chunks.to_parquet(out_dir / CHUNKS_FILE, index=False)
    for name in [LISTINGS_FILE, POIS_FILE]:
        shutil.copyfile(processed / name, out_dir / name)

    report_path = processed / "ingest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    spec = getModel(model_key)
    manifest = {
        "version": version,
        "city": city,
        "snapshot_date": CITY_CONFIG[city]["snapshot"],
        "sources": manifestSources(report),
        "embedding": {
            "key": spec.key, "name": spec.name, "dim": spec.dim,
            "query_prefix": spec.query_prefix, "passage_prefix": spec.passage_prefix,
            "normalized": True, "max_seq_length": MAX_SEQ_LENGTH,
        },
        "chunking": {"max_chars": max_chars, "min_chars": MIN_CHARS,
                     "split": "sentence windows, hard split fallback"},
        "limit_listings": limit,
        "counts": {
            "listings": len(listings),
            "chunks": len(chunks),
            "by_source": chunks["source"].value_counts().to_dict(),
            "by_field": chunks["field"].value_counts().to_dict(),
        },
        "faiss_ntotal": index.ntotal,
        "files": {name: {"bytes": (out_dir / name).stat().st_size,
                         "sha256": sha256File(out_dir / name)}
                  for name in [INDEX_FILE, CHUNKS_FILE, LISTINGS_FILE, POIS_FILE]},
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": gitCommit(),
        "build_seconds": round(time.time() - started, 1),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    return manifest


# ---------- model benchmark (decision support) ----------

# query -> start of the review texts that should rank high (fixture mini_porto)
SANITY = {
    "the wifi kept disconnecting during calls": ["El wifi no funcionaba"],
    "fast internet good for video calls": ["Le wifi était très rapide", "Internet excelente"],
    "very quiet apartment at night": ["Apartamento muito silencioso", "Appartement très calme"],
    "noisy street, bars": ["Muy ruidoso"],
    "there is a gym near the apartment": ["Há um ginásio"],
    "good desk to work from": ["Perfect for remote work"],  # English control
}


def fixtureReviewChunks() -> pd.DataFrame:
    snapshot = CITY_CONFIG["porto"]["snapshot"]
    listings = normalizeListings(readListingsCsv(FIXTURE_DIR / "listings.csv"), "porto", snapshot)
    raw = pd.read_csv(FIXTURE_DIR / "reviews.csv", dtype=str, keep_default_na=False)
    reviews = normalizeReviews(raw, set(listings["listing_id"]))
    chunks = buildChunks(listings, reviews)
    return chunks[chunks["source"] == "review"].reset_index(drop=True)


def sanityRanks(key: str, chunks: pd.DataFrame) -> list[dict]:
    texts = chunks["text"].tolist()
    vectors = embedTexts(texts, key, "passage")
    rows = []
    for query, expected in SANITY.items():
        scores = vectors @ embedTexts([query], key, "query")[0]
        order = np.argsort(-scores)
        for start in expected:
            position = next(i for i, t in enumerate(texts) if t.startswith(start))
            rows.append({"query": query, "expected": start,
                         "rank": int(np.where(order == position)[0][0]) + 1})
    return rows


def benchmark(city: str = "porto", sample: int = 2000, batch_size: int = 64) -> list[dict]:
    reviews = pd.read_parquet(processedDir(city) / "reviews.parquet")
    texts = reviews["comments"].sample(sample, random_state=0).tolist()
    fixture = fixtureReviewChunks()

    results = []
    for key in EMBEDDING_MODELS:
        loadModel(key)  # the first load may download the model
        loadModel.cache_clear()
        started = time.time()
        loadModel(key)
        load_s = time.time() - started

        embedTexts(texts[:batch_size], key)  # warm-up
        started = time.time()
        embedTexts(texts, key, batch_size=batch_size)
        speed = sample / (time.time() - started)

        ranks = sanityRanks(key, fixture)
        results.append({"model": key, "load_s": round(load_s, 1), "chunks_per_s": round(speed),
                        "ranks": ranks,
                        "mean_rank": round(float(np.mean([r["rank"] for r in ranks])), 1),
                        "top3": sum(r["rank"] <= 3 for r in ranks), "of": len(ranks)})
        loadModel.cache_clear()
    return results


def printBenchmark(results: list[dict]) -> None:
    print(f"{'model':22} {'load s':>7} {'chunks/s':>9} {'mean rank':>10} {'in top3':>8}")
    for r in results:
        print(f"{r['model']:22} {r['load_s']:>7} {r['chunks_per_s']:>9} "
              f"{r['mean_rank']:>10} {r['top3']:>5}/{r['of']}")
    print("\nrank of each expected review (1 = best):")
    for i, row in enumerate(results[0]["ranks"]):
        cells = "  ".join(f"{r['model']}={r['ranks'][i]['rank']}" for r in results)
        print(f"  {row['query'][:42]:42} -> {row['expected'][:26]:26} {cells}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the v2 chunk index artifacts")
    parser.add_argument("--city", default="porto", choices=sorted(CITY_CONFIG))
    parser.add_argument("--model", choices=sorted(EMBEDDING_MODELS))
    parser.add_argument("--version", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="only the first N listings (debug)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per model file otherwise

    if args.benchmark:
        printBenchmark(benchmark(args.city))
        return
    if not args.model:
        parser.error("--model is required")
    if args.limit and not args.out_dir:
        parser.error("--limit builds a partial index: pass --out-dir so v<version> is not overwritten")

    out_dir = args.out_dir or refinedDir(args.city, args.version)
    manifest = buildArtifacts(processedDir(args.city), out_dir, args.city, args.model,
                              args.version, args.limit)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
