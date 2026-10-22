"""
Points of interest from OpenStreetMap (gyms) and distance from each stay.

    python -m ingest.osm --city porto

Run after ingest.insideairbnb: it adds nearest_gym_m / nearest_gym_name to
listings.parquet. Caveats: Inside Airbnb anonymizes coordinates up to ~150 m,
and distances are straight-line (haversine), not walking routes.
Data: © OpenStreetMap contributors, ODbL.
"""

import argparse
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from constants import CITY_CONFIG, RAW_DIR, processedDir
from ingest.insideairbnb import readListings, updateReport, writeListings

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
USER_AGENT = "NomadStay-ingest/0.1 (offline gym lookup for a portfolio project)"
ATTRIBUTION = "© OpenStreetMap contributors, ODbL"
EARTH_RADIUS_M = 6371008.8


def gymQuery(bbox: tuple) -> str:
    area = ",".join(str(x) for x in bbox)  # south, west, north, east
    return (
        "[out:json][timeout:60];("
        f"nwr[leisure=fitness_centre]({area});"
        f'nwr[leisure=sports_centre][sport~"fitness"]({area});'
        f"nwr[amenity=gym]({area});"
        ");out center;"
    )


def fetchPois(city: str, today: date | None = None) -> tuple[dict, Path]:
    """Gyms in the city bbox as raw Overpass JSON. Reuses today's saved file."""
    today = today or date.today()
    path = RAW_DIR / city / f"osm_gyms_{today.isoformat()}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path

    response = httpx.post(OVERPASS_URL, data={"data": gymQuery(CITY_CONFIG[city]["bbox"])},
                          headers={"User-Agent": USER_AGENT}, timeout=90)
    response.raise_for_status()
    raw = response.json()
    # Overpass answers 200 with a "remark" when the query timed out
    if "elements" not in raw or "runtime error" in raw.get("remark", ""):
        raise RuntimeError(f"Overpass query failed: {raw.get('remark', 'no elements')}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return raw, path


def parsePois(raw: dict, kind: str = "gym") -> pd.DataFrame:
    """Overpass elements -> osm_id ("node/1", "way/2"), kind, name, lat, lon."""
    rows = {}
    for e in raw.get("elements", []):
        point = e if "lat" in e else e.get("center", {})  # ways/relations carry a center
        if "lat" not in point or "lon" not in point:
            continue
        osm_id = f"{e['type']}/{e['id']}"
        rows[osm_id] = {
            "osm_id": osm_id, "kind": kind, "name": e.get("tags", {}).get("name"),
            "lat": float(point["lat"]), "lon": float(point["lon"]),
        }
    return pd.DataFrame(rows.values(), columns=["osm_id", "kind", "name", "lat", "lon"])


def addNearestPoi(listings: pd.DataFrame, pois: pd.DataFrame, kind: str = "gym") -> pd.DataFrame:
    """
    Add nearest_<kind>_m (meters, rounded) and nearest_<kind>_name to listings.
    Straight-line haversine distance; listing coordinates are anonymized by up
    to ~150 m, so treat small distances as approximate.
    """
    out = listings.copy()
    near = pois[pois["kind"] == kind].reset_index(drop=True)
    if near.empty:
        out[f"nearest_{kind}_m"] = np.nan
        out[f"nearest_{kind}_name"] = None
        return out

    tree = BallTree(np.radians(near[["lat", "lon"]].to_numpy(float)), metric="haversine")
    query = np.radians(out[["latitude", "longitude"]].to_numpy(float))
    dist, idx = tree.query(query, k=1)
    out[f"nearest_{kind}_m"] = np.round(dist[:, 0] * EARTH_RADIUS_M)
    names = near["name"].to_numpy(object)[idx[:, 0]]
    out[f"nearest_{kind}_name"] = [n if isinstance(n, str) else None for n in names]
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Add nearest gym to ingested listings")
    parser.add_argument("--city", default="porto", choices=sorted(CITY_CONFIG))
    args = parser.parse_args(argv)

    out_dir = processedDir(args.city)
    raw, raw_path = fetchPois(args.city)
    pois = parsePois(raw)
    pois.to_parquet(out_dir / "pois.parquet", index=False)

    listings = addNearestPoi(readListings(out_dir / "listings.parquet"), pois, "gym")
    writeListings(listings, out_dir / "listings.parquet")

    report = {
        "attribution": ATTRIBUTION,
        "raw_file": raw_path.name,
        "fetched_date": raw_path.stem.removeprefix("osm_gyms_"),
        "osm_data_timestamp": raw.get("osm3s", {}).get("timestamp_osm_base"),
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "bbox": CITY_CONFIG[args.city]["bbox"],
        "gyms": len(pois),
        "gyms_without_name": int(pois["name"].isna().sum()),
        "distance": "straight-line haversine; listing coordinates anonymized up to ~150 m",
    }
    updateReport(out_dir / "ingest_report.json", "osm", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
