"""
Packages the serving set the API loads: an index set (ingest/build.py) plus
the evidence table built from the precomputed stances (ingest/stance.py).
The index is reused as is, so no re-embedding.

    python -m ingest.package --city porto --from-version 2 --version 3 [--upload]

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
from ingest.build import gitCommit
from ingest.stance import evidenceTable
from retrieval.chunk_index import LISTINGS_FILE, MANIFEST, sha256File
from retrieval.evidence import EVIDENCE_FILE, SERVING_FILES

logger = logging.getLogger(__name__)


def packageServingSet(index_dir: Path, processed: Path, out_dir: Path, version: int) -> dict:
    """index set + evidence table -> serving set in out_dir. Returns the manifest."""
    source = json.loads((index_dir / MANIFEST).read_text(encoding="utf-8"))
    evidence = evidenceTable(pd.read_parquet(processed / "stances.parquet"),
                             pd.read_parquet(processed / "reviews.parquet"))

    listing_ids = set(pd.read_parquet(index_dir / LISTINGS_FILE, columns=["listing_id"])["listing_id"])
    unknown = set(evidence["listing_id"]) - listing_ids
    if unknown:
        raise ValueError(f"evidence for {len(unknown)} listings not in the index set, e.g. {sorted(unknown)[:3]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in SERVING_FILES:
        if name not in (EVIDENCE_FILE, MANIFEST):
            shutil.copyfile(index_dir / name, out_dir / name)
    evidence.to_parquet(out_dir / EVIDENCE_FILE, index=False)

    report_path = processed / "ingest_report.json"
    stance_report = (json.loads(report_path.read_text(encoding="utf-8")).get("stance", {})
                     if report_path.exists() else {})
    cited = evidence["stance"].isin(["supports", "contradicts"])
    manifest = {
        **source,
        "version": version,
        "based_on": {"version": source["version"], "git_commit": source.get("git_commit")},
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
    parser.add_argument("--version", type=int, default=3)
    parser.add_argument("--upload", action="store_true", help="upload the set to S3 afterwards")
    args = parser.parse_args(argv)

    out_dir = refinedDir(args.city, args.version)
    manifest = packageServingSet(refinedDir(args.city, args.from_version), processedDir(args.city),
                                 out_dir, args.version)
    print(json.dumps({k: manifest[k] for k in ("version", "based_on", "evidence", "counts")}, indent=2))

    if args.upload:
        from storage.s3 import uploadArtifacts
        for key in uploadArtifacts(out_dir, args.version, args.city, SERVING_FILES):
            print("uploaded", key)


if __name__ == "__main__":
    main()
