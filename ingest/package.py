"""
Packages the serving set the API loads: an index set (ingest/build.py) plus
the evidence table built from the precomputed stances (ingest/stance.py).
The index is reused as is, so no re-embedding.

Map data (nearest gym / metro station, pois.parquet) comes from processed/,
so a new `python -m ingest.osm` reaches the API without a rebuild. Only the
map columns are taken: everything the chunks were made from stays as built.

    python -m ingest.package --city porto --from-version 2 --version 4 [--upload]

--upload sends the set to s3://$BUCKET/rag/v{version}/{city}/ (manifest last).
"""

import argparse
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from constants import processedDir, refinedDir
from domain.schemas import Listing
from ingest.build import gitCommit, manifestSources
from ingest.stance import evidenceTable
from retrieval.chunk_index import LISTINGS_FILE, MANIFEST, POIS_FILE, sha256File
from retrieval.evidence import EVIDENCE_FILE, SERVING_FILES

logger = logging.getLogger(__name__)

# Written by ingest.osm; nothing in the chunks depends on them
MAP_COLUMNS = ["nearest_gym_m", "nearest_gym_name", "nearest_metro_m", "nearest_metro_name"]


def withFreshMapData(built: pd.DataFrame, processed: pd.DataFrame) -> pd.DataFrame:
    """
    The index set's listings (raw parquet columns) with the map columns of
    processed/. Every built listing must be there, with the map columns.
    """
    missing_columns = [c for c in MAP_COLUMNS if c not in processed.columns]
    if missing_columns:
        raise ValueError(f"processed listings have no {missing_columns}: run `python -m ingest.osm` first")
    fresh = processed.set_index("listing_id")[MAP_COLUMNS]
    missing = set(built["listing_id"]) - set(fresh.index)
    if missing:
        raise ValueError(f"{len(missing)} listings of the index set are not in processed/, "
                         f"e.g. {sorted(missing)[:3]}: rebuild the index set")
    out = built.drop(columns=MAP_COLUMNS, errors="ignore").join(fresh, on="listing_id")
    return out[list(Listing.model_fields)]  # schema order, as ingest writes it


def packageServingSet(index_dir: Path, processed: Path, out_dir: Path, version: int) -> dict:
    """index set + evidence table + fresh map data -> serving set in out_dir. Returns the manifest."""
    source = json.loads((index_dir / MANIFEST).read_text(encoding="utf-8"))
    evidence = evidenceTable(pd.read_parquet(processed / "stances.parquet"),
                             pd.read_parquet(processed / "reviews.parquet"))

    listings = withFreshMapData(pd.read_parquet(index_dir / LISTINGS_FILE),
                                pd.read_parquet(processed / LISTINGS_FILE))
    unknown = set(evidence["listing_id"]) - set(listings["listing_id"])
    if unknown:
        raise ValueError(f"evidence for {len(unknown)} listings not in the index set, e.g. {sorted(unknown)[:3]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SERVING_FILES:
        if name not in (EVIDENCE_FILE, MANIFEST, LISTINGS_FILE, POIS_FILE):
            shutil.copyfile(index_dir / name, out_dir / name)
    listings.to_parquet(out_dir / LISTINGS_FILE, index=False)
    shutil.copyfile(processed / POIS_FILE, out_dir / POIS_FILE)
    evidence.to_parquet(out_dir / EVIDENCE_FILE, index=False)

    report_path = processed / "ingest_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    stance_report = report.get("stance", {})
    pois = pd.read_parquet(out_dir / POIS_FILE)
    cited = evidence["stance"].isin(["supports", "contradicts"])
    manifest = {
        **source,
        "version": version,
        "based_on": {"version": source["version"], "git_commit": source.get("git_commit")},
        "sources": {**source.get("sources", {}),
                    "openstreetmap": manifestSources(report)["openstreetmap"]},
        "map_data": {
            "from": "processed (ingest.osm), joined on listing_id",
            "pois_by_kind": pois["kind"].value_counts().to_dict(),
            "listings_with": {c: int(listings[c].notna().sum()) for c in MAP_COLUMNS if c.endswith("_m")},
        },
        "evidence": {
            "judge_model": stance_report.get("model"),
            "guidelines_version": stance_report.get("guidelines_version"),
            "pairs": len(evidence),
            "with_quote": int((cited & evidence["quote"].notna()).sum()),
            "by_stance": evidence["stance"].fillna("missing").value_counts().to_dict(),
        },
        "files": {name: {"bytes": (out_dir / name).stat().st_size, "sha256": sha256File(out_dir / name)}
                  for name in SERVING_FILES if name != MANIFEST},
        "packaged_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": gitCommit(),
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Package the serving set (index set + evidence)")
    parser.add_argument("--city", default="porto")
    parser.add_argument("--from-version", type=int, default=2)
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--upload", action="store_true", help="upload the set to S3 afterwards")
    args = parser.parse_args(argv)

    out_dir = refinedDir(args.city, args.version)
    manifest = packageServingSet(refinedDir(args.city, args.from_version), processedDir(args.city),
                                 out_dir, args.version)
    print(json.dumps({k: manifest[k] for k in ("version", "based_on", "map_data", "evidence", "counts")},
                     indent=2))

    if args.upload:
        from storage.s3 import uploadArtifacts
        for key in uploadArtifacts(out_dir, args.version, args.city, SERVING_FILES):
            print("uploaded", key)


if __name__ == "__main__":
    main()
