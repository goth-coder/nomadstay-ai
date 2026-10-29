"""
Offline ingestion of an Inside Airbnb snapshot (listings + reviews).

    python -m ingest.insideairbnb --city porto

Writes data/processed/<city>/<snapshot>/{listings,reviews}.parquet and
ingest_report.json. Listings follow domain.schemas.Listing. Reviews are
cleaned, filtered and sampled (see sampleReviews). Reviewer id/name are never
read (personal data). Data: Inside Airbnb, CC BY 4.0.
"""

import argparse
import hashlib
import html
import json
import logging
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
from pydantic import ValidationError

from constants import CITY_CONFIG, processedDir, rawDir
from domain.aspects import ASPECTS, claimedAspects, mentionedAspects, parseWifiMbps
from domain.schemas import Listing

logger = logging.getLogger(__name__)

LISTING_COLUMNS = [
    "id", "listing_url", "last_scraped", "name", "description",
    "neighborhood_overview", "neighbourhood_cleansed", "neighbourhood_group_cleansed",
    "latitude", "longitude", "property_type", "room_type", "accommodates",
    "amenities", "price", "price_quote_price_per_night",
    "number_of_reviews", "last_review", "review_scores_rating",
]
REVIEW_COLUMNS = ["listing_id", "id", "date", "comments"]  # no reviewer_id/name

MIN_COMMENT_CHARS = 15
MAX_INVALID_SHARE = 0.01  # more invalid listings than this means the format changed

BR_TAG = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)
HTML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")  # needs a letter, so "<3" or "a < b" survive
CUT_TAG = re.compile(r"</?[a-zA-Z][^<>]*$")  # text truncated mid-tag, e.g. "...<br /"


# ---------- download ----------

def downloadFile(url: str, dest: Path) -> bool:
    """Download url to dest through a .part file. Returns False if it already existed."""
    if dest.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with open(part, "wb") as f:
            f.writelines(response.iter_bytes(1 << 20))
    part.replace(dest)
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


# ---------- cleaning helpers ----------

def cleanText(text: str, keep_newlines: bool = True) -> str:
    """Strip HTML tags, unescape entities and collapse whitespace."""
    if not text:
        return ""
    text = BR_TAG.sub("\n", text)
    text = CUT_TAG.sub("", HTML_TAG.sub(" ", text))
    text = html.unescape(text).replace("\xa0", " ")
    if keep_newlines:
        text = re.sub(r"[ \t\r\f\v]+", " ", text)
        text = re.sub(r" ?\n ?", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
    else:
        text = re.sub(r"\s+", " ", text)
    return text.strip()


def parsePrice(text: str) -> float | None:
    """'$1,050.00' -> 1050.0. Empty, unparseable or non-positive -> None."""
    cleaned = re.sub(r"[^\d.]", "", text or "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def pickPrice(price: str, quote: str) -> tuple[float | None, str]:
    # listed price first, then the quote; never guess
    if (value := parsePrice(price)) is not None:
        return value, "listed"
    if (value := parsePrice(quote)) is not None:
        return value, "quote"
    return None, "missing"


def parseAmenities(text: str) -> list[str]:
    try:
        items = json.loads(text) if text else []
    except json.JSONDecodeError:
        return []
    return [a.strip() for a in items if isinstance(a, str) and a.strip()] if isinstance(items, list) else []


def optionalInt(text: str) -> int | None:
    return int(float(text)) if text else None


# ---------- listings ----------

def listingRecord(row: dict, city: str, snapshot: str) -> dict:
    """One raw CSV row -> validated Listing as a dict. Raises on invalid data."""
    amenities = parseAmenities(row["amenities"])
    price, status = pickPrice(row["price"], row["price_quote_price_per_night"])
    listing = Listing.model_validate({
        "listing_id": row["id"].strip(),
        "name": cleanText(row["name"], keep_newlines=False),
        "url": row["listing_url"],
        "city": city,
        "neighbourhood": row["neighbourhood_cleansed"] or None,
        "neighbourhood_group": row["neighbourhood_group_cleansed"] or None,
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "property_type": row["property_type"] or None,
        "room_type": row["room_type"] or None,
        "accommodates": optionalInt(row["accommodates"]),
        "price": price,
        "currency": CITY_CONFIG[city]["currency"],
        "price_status": status,
        "amenities": amenities,
        "claims": claimedAspects(amenities),
        "wifi_mbps": parseWifiMbps(amenities),
        "rating": float(row["review_scores_rating"]) if row["review_scores_rating"] else None,
        "number_of_reviews": optionalInt(row["number_of_reviews"]) or 0,
        "last_review": row["last_review"] or None,
        "description": cleanText(row["description"]),
        "neighborhood_overview": cleanText(row["neighborhood_overview"]) or None,
        "snapshot_date": snapshot,
    })
    return listing.model_dump()


def normalizeListings(df: pd.DataFrame, city: str, snapshot: str) -> pd.DataFrame:
    """
    Raw listings (all columns as str, no NaN) -> one row per Listing field.

    Keeps only the city's municipality. Rows that fail Listing validation are
    dropped and reported in df.attrs["invalid"] (count + examples); a run with
    more than MAX_INVALID_SHARE invalid rows fails, since that means the CSV
    format changed. One odd host row should not block the whole city.
    """
    group = CITY_CONFIG[city]["neighbourhood_group"].casefold()
    inside = df[df["neighbourhood_group_cleansed"].str.strip().str.casefold() == group]

    records, invalid = [], []
    for row in inside.to_dict("records"):
        try:
            records.append(listingRecord(row, city, snapshot))
        except (ValueError, ValidationError, KeyError) as e:
            invalid.append({"id": row.get("id"), "error": str(e).splitlines()[0][:200]})

    if len(invalid) > MAX_INVALID_SHARE * max(len(inside), 1):
        raise ValueError(f"{len(invalid)} of {len(inside)} listings are invalid, e.g. {invalid[:3]}")

    out = pd.DataFrame(records, columns=list(Listing.model_fields))
    # nullable ints keep None as <NA> instead of turning the column into float
    for column in ["accommodates", "wifi_mbps"]:
        out[column] = out[column].astype("Int64")
    out["price"] = out["price"].astype("float64")
    out["rating"] = out["rating"].astype("float64")
    out["nearest_gym_m"] = out["nearest_gym_m"].astype("float64")
    out["nearest_metro_m"] = out["nearest_metro_m"].astype("float64")
    out.attrs["invalid"] = {"count": len(invalid), "examples": invalid[:5]}
    out.attrs["total_rows"] = len(df)
    return out


def readListingsCsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, usecols=LISTING_COLUMNS, dtype=str, keep_default_na=False)


# ---------- reviews ----------

def normalizeReviews(df: pd.DataFrame, listing_ids: set[str]) -> pd.DataFrame:
    """
    Raw reviews -> reviews of the kept listings with clean comments.
    Drops duplicates, empty and very short comments (< MIN_COMMENT_CHARS) and
    adds `aspects` (list of aspect names the text mentions).
    """
    df = df[df["listing_id"].isin(listing_ids)]
    out = pd.DataFrame({
        "listing_id": df["listing_id"].astype(str),
        "review_id": df["id"].astype(str),
        "date": pd.to_datetime(df["date"], errors="coerce").dt.date,
        "comments": pd.Series([cleanText(c, keep_newlines=False) for c in df["comments"]],
                             index=df.index, dtype="str"),
    })
    out = out[out["comments"].str.len() >= MIN_COMMENT_CHARS].drop_duplicates("review_id")
    out["aspects"] = [sorted(mentionedAspects(c)) for c in out["comments"]]
    return out.reset_index(drop=True)


def sampleReviews(reviews: pd.DataFrame, per_listing: int = 20,
                  keep_aspects: set[str] | None = None) -> pd.DataFrame:
    """
    Keep the `per_listing` most recent reviews of each listing PLUS every review
    that mentions an aspect, so rare mentions (a wifi complaint from 2019)
    survive the cap. keep_aspects limits which aspects count (None = all);
    broad ones like quiet/transport match a third of all reviews.
    Result is ordered by listing, newest first.
    """
    ordered = reviews.assign(_date=pd.to_datetime(reviews["date"])).sort_values(
        ["listing_id", "_date", "review_id"], ascending=[True, False, False], kind="stable"
    )
    rank = ordered.groupby("listing_id").cumcount()
    wanted = keep_aspects if keep_aspects is not None else set(ASPECTS)
    mentions = ordered["aspects"].map(lambda names: bool(wanted.intersection(names)))
    kept = ordered[(rank < per_listing) | mentions]
    return kept.drop(columns="_date").reset_index(drop=True)


def readReviewsCsv(path: Path, listing_ids: set[str], chunk_rows: int = 200_000):
    """Stream the big reviews CSV; returns (total_rows, normalized reviews of kept listings)."""
    total, parts = 0, []
    reader = pd.read_csv(path, usecols=REVIEW_COLUMNS, dtype=str, keep_default_na=False,
                         chunksize=chunk_rows)
    for chunk in reader:
        total += len(chunk)
        parts.append(normalizeReviews(chunk, listing_ids))
    if not parts:
        parts = [normalizeReviews(pd.DataFrame(columns=REVIEW_COLUMNS, dtype=str), listing_ids)]
    return total, pd.concat(parts, ignore_index=True)


# ---------- parquet io ----------

def writeListings(listings: pd.DataFrame, path: Path) -> None:
    # claims is a dict with varying keys (and may be empty): stored as JSON text
    out = listings.copy()
    out["claims"] = out["claims"].map(lambda c: json.dumps(c, ensure_ascii=False))
    out.to_parquet(path, index=False)


def readListings(path: Path) -> pd.DataFrame:
    out = pd.read_parquet(path)
    out["amenities"] = out["amenities"].map(list)  # parquet lists come back as numpy arrays
    out["claims"] = out["claims"].map(json.loads)
    return out


def countContaining(series: pd.Series, item: str) -> int:
    return int(series.map(lambda values: item in values).sum())


def updateReport(path: Path, section: str, data: dict) -> None:
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report[section] = data
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- pipeline ----------

def ingest(listings_csv: Path, reviews_csv: Path, out_dir: Path, city: str,
           snapshot: str, per_listing: int = 20, keep_aspects: set[str] | None = None) -> dict:
    """Normalize both CSVs into out_dir and return the report written there."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    raw = readListingsCsv(listings_csv)
    listings = normalizeListings(raw, city, snapshot)
    writeListings(listings, out_dir / "listings.parquet")

    total_reviews, reviews = readReviewsCsv(reviews_csv, set(listings["listing_id"]))
    sampled = sampleReviews(reviews, per_listing, keep_aspects)
    sampled.to_parquet(out_dir / "reviews.parquet", index=False)

    aspect_names = sorted({a for aspects in sampled["aspects"] for a in aspects})
    report = {
        "city": city,
        "snapshot_date": snapshot,
        "license": CITY_CONFIG[city]["license"],
        "source": CITY_CONFIG[city]["base_url"],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "listings_total": len(raw),
        "listings_in_municipality": len(listings) + listings.attrs["invalid"]["count"],
        "listings_invalid": listings.attrs["invalid"],
        "listings_kept": len(listings),
        "price_status": listings["price_status"].value_counts().to_dict(),
        "claims": {a: countContaining(listings["claims"], a)
                   for a in ["wifi", "workspace", "gym_nearby"]},
        "listings_with_wifi_mbps": int(listings["wifi_mbps"].notna().sum()),
        "reviews_total": total_reviews,
        "reviews_after_filters": len(reviews),
        "reviews_sampled": len(sampled),
        "reviews_per_listing_cap": per_listing,
        "reviews_keep_aspects": sorted(keep_aspects if keep_aspects is not None else ASPECTS),
        "sampled_reviews_by_aspect": {
            a: countContaining(sampled["aspects"], a) for a in aspect_names},
        "raw_files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                      for p in [listings_csv, reviews_csv]},
        "seconds": round(time.time() - started, 1),
    }
    updateReport(out_dir / "ingest_report.json", "insideairbnb", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Ingest an Inside Airbnb snapshot")
    parser.add_argument("--city", default="porto", choices=sorted(CITY_CONFIG))
    parser.add_argument("--reviews-per-listing", type=int, default=20)
    parser.add_argument("--keep-aspects", default=None,
                        help="comma list of aspects whose mentions bypass the cap (default: all)")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    keep_aspects = set(args.keep_aspects.split(",")) if args.keep_aspects else None
    if keep_aspects and not keep_aspects <= set(ASPECTS):
        parser.error(f"unknown aspects: {sorted(keep_aspects - set(ASPECTS))}")

    config = CITY_CONFIG[args.city]
    raw_dir = rawDir(args.city)
    paths = {name: raw_dir / name for name in ["listings.csv.gz", "reviews.csv.gz"]}
    if not args.skip_download:
        for name, path in paths.items():
            if downloadFile(f"{config['base_url']}/{name}", path):
                print(f"downloaded {path}")

    report = ingest(paths["listings.csv.gz"], paths["reviews.csv.gz"], processedDir(args.city),
                    args.city, config["snapshot"], args.reviews_per_listing, keep_aspects)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
