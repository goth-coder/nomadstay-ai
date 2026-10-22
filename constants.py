from pathlib import Path

# Per-city Inside Airbnb snapshot + OSM bounding box
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
ARTIFACT_DIR = Path("data/refined")

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

# What the API serves: rag/v{ARTIFACT_VERSION}/{CITY}/ (local copy or S3)
CITY = "porto"
ARTIFACT_VERSION = 3  # v2 index + chunks + listings + pois, plus evidence.parquet
EMBEDDING_MODEL = "multilingual-minilm"


def rawDir(city: str) -> Path:
    return RAW_DIR / city / CITY_CONFIG[city]["snapshot"]


def processedDir(city: str) -> Path:
    return PROCESSED_DIR / city / CITY_CONFIG[city]["snapshot"]


def refinedDir(city: str, version: int) -> Path:
    # Artifact set the API loads (and S3 mirrors): index + chunks + listings + manifest
    return ARTIFACT_DIR / f"v{version}" / city
