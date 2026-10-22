from pathlib import Path

ARTIFACT_DIR = Path("data/refined")
INDEX_PATH = ARTIFACT_DIR / "hotels.index"
DOCS_PATH = ARTIFACT_DIR / "hotels_doc.json"


# v2 (NomadStay): per-city Inside Airbnb snapshot + OSM bounding box
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

CITY_CONFIG = {
    "porto": {
        "snapshot": "2026-09-19",
        "base_url": "https://data.insideairbnb.com/portugal/norte/porto/2026-09-19/data",
        "neighbourhood_group": "PORTO",  # municipality only (excludes Gaia, Matosinhos...)
        "currency": "EUR",  # Inside Airbnb shows "$" but prices are local currency
        "license": "CC BY 4.0",
        "bbox": (41.10, -8.72, 41.20, -8.54),  # south, west, north, east; margin around the municipality
    },
}


def rawDir(city: str) -> Path:
    return RAW_DIR / city / CITY_CONFIG[city]["snapshot"]


def processedDir(city: str) -> Path:
    return PROCESSED_DIR / city / CITY_CONFIG[city]["snapshot"]


def refinedDir(city: str, version: int) -> Path:
    # Artifact set the API loads (and S3 mirrors): index + chunks + listings + manifest
    return ARTIFACT_DIR / f"v{version}" / city
