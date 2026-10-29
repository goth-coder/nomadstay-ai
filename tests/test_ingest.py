import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from domain.schemas import Listing
from ingest import insideairbnb, osm
from ingest.insideairbnb import (
    cleanText,
    ingest,
    normalizeListings,
    normalizeReviews,
    parsePrice,
    pickPrice,
    readListings,
    readListingsCsv,
    sampleReviews,
)

FIXTURES = Path(__file__).parent / "fixtures"
MINI = FIXTURES / "mini_porto"
EXPECTED = json.loads((MINI / "expected.json").read_text(encoding="utf-8"))
SNAPSHOT = EXPECTED["snapshot_date"]
IN_PORTO = {k: v for k, v in EXPECTED["listings"].items() if v["in_porto"]}


@pytest.fixture(scope="module")
def listings():
    return normalizeListings(readListingsCsv(MINI / "listings.csv"), "porto", SNAPSHOT)


@pytest.fixture(scope="module")
def reviews(listings):
    raw = pd.read_csv(MINI / "reviews.csv", dtype=str, keep_default_na=False)
    return normalizeReviews(raw, set(listings["listing_id"]))


def row(listings, listing_id):
    return listings[listings["listing_id"] == listing_id].iloc[0]


# ---------- prices ----------

def test_parsePrice_handles_currency_symbol_and_thousands():
    assert parsePrice("$1,050.00") == 1050.0
    assert parsePrice("$85.00") == 85.0
    assert parsePrice("") is None
    assert parsePrice("$0.00") is None


def test_pickPrice_listed_quote_missing():
    assert pickPrice("$80.00", "45.00") == (80.0, "listed")
    assert pickPrice("", "45.00") == (45.0, "quote")
    assert pickPrice("", "") == (None, "missing")


def test_fixture_prices(listings):
    assert row(listings, "1052141614265835101")["price"] == 1050.0
    quote = row(listings, "40001008")
    assert (quote["price"], quote["price_status"]) == (45.0, "quote")
    missing = row(listings, "40001009")
    assert pd.isna(missing["price"]) and missing["price_status"] == "missing"
    assert (listings["currency"] == "EUR").all()


# ---------- listings ----------

def test_only_porto_municipality_is_kept(listings):
    assert set(listings["listing_id"]) == set(IN_PORTO)
    assert "40001010" not in set(listings["listing_id"])  # Gaia
    assert "40001011" not in set(listings["listing_id"])  # Matosinhos
    assert (listings["neighbourhood_group"] == "PORTO").all()


def test_ids_are_strings_including_19_digit(listings):
    assert all(isinstance(i, str) for i in listings["listing_id"])
    assert "1436961951661730861" in set(listings["listing_id"])


def test_columns_match_listing_schema(listings):
    assert list(listings.columns) == list(Listing.model_fields)


def test_every_row_validates_against_schema(listings):
    for rec in listings.to_dict("records"):
        rec = {k: None if not isinstance(v, (list, dict)) and pd.isna(v) else v
               for k, v in rec.items()}
        Listing.model_validate(rec)
    assert listings.attrs["invalid"]["count"] == 0


def test_claims_and_wifi_match_expected(listings):
    for listing_id, want in IN_PORTO.items():
        got = row(listings, listing_id)
        assert sorted(got["claims"]) == sorted(want["claims"]), want["name"]
        if "wifi_mbps" in want:
            assert got["wifi_mbps"] == want["wifi_mbps"]
        else:
            assert pd.isna(got["wifi_mbps"])


def test_rating_is_none_when_empty(listings):
    assert pd.isna(row(listings, "40001020")["rating"])
    assert row(listings, "1436961951661730861")["rating"] == 4.92


def test_html_stripped_from_description(listings):
    description = row(listings, "1052141614265835101")["description"]
    assert "<" not in description and ">" not in description
    assert description == "Fiber internet.\n\nStanding desk and monitor.\nQuiet building."


def test_cleanText():
    assert cleanText("a<br />b<BR/>c") == "a\nb\nc"
    assert cleanText("Tom &amp; Jerry &lt;3") == "Tom & Jerry <3"
    assert cleanText("I <3 it, 2 < 5 ok") == "I <3 it, 2 < 5 ok"  # not tags
    assert cleanText("see you<br /") == "see you"  # text cut mid-tag
    assert cleanText("a <b>bold</b>  \n\n\n\n c", keep_newlines=False) == "a bold c"


def test_invalid_rows_are_dropped_and_reported(monkeypatch):
    monkeypatch.setattr(insideairbnb, "MAX_INVALID_SHARE", 0.1)  # the fixture is tiny
    raw = readListingsCsv(MINI / "listings.csv")
    raw.loc[raw["id"] == "32235516", "latitude"] = "not-a-number"
    out = normalizeListings(raw, "porto", SNAPSHOT)
    assert out.attrs["invalid"]["count"] == 1
    assert out.attrs["invalid"]["examples"][0]["id"] == "32235516"
    assert "32235516" not in set(out["listing_id"])


def test_too_many_invalid_rows_fail_loudly():
    raw = readListingsCsv(MINI / "listings.csv")
    raw["longitude"] = "x"
    with pytest.raises(ValueError, match="invalid"):
        normalizeListings(raw, "porto", SNAPSHOT)


# ---------- reviews ----------

def test_reviews_only_of_kept_listings_and_no_personal_data(reviews, listings):
    assert set(reviews["listing_id"]) <= set(listings["listing_id"])
    assert "40001010" not in set(reviews["listing_id"])
    assert list(reviews.columns) == ["listing_id", "review_id", "date", "comments", "aspects"]
    assert all(isinstance(i, str) for i in reviews["review_id"])


def test_review_html_stripped(reviews):
    assert not reviews["comments"].str.contains("<", regex=False).any()
    first = reviews[reviews["review_id"] == "7000000000000000001"].iloc[0]
    assert first["comments"].endswith("Obrigado!")


def test_short_and_empty_comments_dropped():
    raw = pd.DataFrame({
        "listing_id": ["1"] * 4,
        "id": ["a", "b", "c", "d"],
        "date": ["2026-01-01"] * 4,
        "comments": ["", "ok", "<br/>  <br/>", "A perfectly fine long comment."],
    })
    out = normalizeReviews(raw, {"1"})
    assert out["review_id"].tolist() == ["d"]


def test_aspects_column(reviews):
    wifi = reviews[reviews["review_id"] == "7000000000000000001"].iloc[0]
    assert "wifi" in wifi["aspects"]
    filler = reviews[reviews["comments"] == "Great location and a very friendly host."]
    assert (filler["aspects"].map(len) == 0).all()


def test_sampling_keeps_aspect_mentions_beyond_the_cap(reviews):
    # Baixa Classic Flat: 2 old wifi reviews + 6 filler; cap of 2 keeps 2 filler + 2 wifi
    all_reviews = reviews[reviews["listing_id"] == "40001017"]
    assert len(all_reviews) == 8
    sampled = sampleReviews(all_reviews, per_listing=2)
    assert len(sampled) == 4
    assert sampled["aspects"].map(lambda a: "wifi" in a).sum() == 2
    assert sampled["date"].is_monotonic_decreasing  # newest first


def test_sampling_caps_reviews_without_mentions(reviews):
    no_mentions = reviews[reviews["aspects"].map(len) == 0]
    sampled = sampleReviews(no_mentions, per_listing=3)
    assert sampled.groupby("listing_id").size().max() == 3


def test_sampling_keep_aspects_limits_what_bypasses_cap(reviews):
    # wifi, workspace and quiet mentions + 6 filler
    one = reviews[reviews["listing_id"] == "1436961951661730861"]
    only_wifi = sampleReviews(one, per_listing=1, keep_aspects={"wifi"})
    everything = sampleReviews(one, per_listing=1)
    assert len(only_wifi) < len(everything)


def test_sampling_keeps_newest_first(reviews):
    one = reviews[reviews["listing_id"] == "1436961951661730861"]
    sampled = sampleReviews(one, per_listing=2, keep_aspects=set())
    assert list(sampled["review_id"]) == ["7000000000000000001", "7000000000000000002"]


# ---------- end to end ----------

def test_ingest_writes_parquet_and_report(tmp_path):
    report = ingest(MINI / "listings.csv", MINI / "reviews.csv", tmp_path, "porto", SNAPSHOT,
                    per_listing=2)
    assert report["listings_total"] == len(EXPECTED["listings"])
    assert report["listings_kept"] == len(IN_PORTO)
    assert report["price_status"] == {"listed": len(IN_PORTO) - 2, "quote": 1, "missing": 1}
    assert report["reviews_sampled"] <= report["reviews_after_filters"] <= report["reviews_total"]

    back = readListings(tmp_path / "listings.parquet")
    assert list(back.columns) == list(Listing.model_fields)
    assert row(back, "1730176987514588989")["claims"]["gym_nearby"] == "Shared gym in building"
    assert row(back, "40001020")["claims"] == {"wifi": "Wifi"}
    assert isinstance(row(back, "1052141614265835101")["amenities"], list)

    stored = pd.read_parquet(tmp_path / "reviews.parquet")
    assert not {"reviewer_id", "reviewer_name"} & set(stored.columns)
    saved = json.loads((tmp_path / "ingest_report.json").read_text())
    assert saved["insideairbnb"]["listings_kept"] == len(IN_PORTO)


# ---------- OSM ----------

def haversineM(lat1, lon1, lat2, lon2):
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((phi2 - phi1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * osm.EARTH_RADIUS_M * math.asin(math.sqrt(a))


@pytest.fixture
def pois():
    raw = json.loads((FIXTURES / "overpass_gyms.json").read_text(encoding="utf-8"))
    return osm.parsePois(raw)


def stays(*coords):
    return pd.DataFrame({"latitude": [c[0] for c in coords], "longitude": [c[1] for c in coords]})


def test_parsePois_node_way_center_and_missing_coordinates(pois):
    # node/1001 and way/1001 share an id: the type keeps them apart
    assert pois["osm_id"].tolist() == ["node/1001", "way/1001", "node/1003"]
    assert (pois["kind"] == "gym").all()
    way = pois[pois["osm_id"] == "way/1001"].iloc[0]
    assert (way["lat"], way["lon"]) == (41.16, -8.62)  # taken from center
    assert way["name"] == "Gym Way"
    assert pd.isna(pois[pois["osm_id"] == "node/1003"].iloc[0]["name"])  # unnamed gym


def test_gymQuery_covers_the_three_tags():
    query = osm.gymQuery((41.1, -8.7, 41.2, -8.5))
    assert "out center" in query
    assert "leisure=fitness_centre" in query and "amenity=gym" in query
    assert 'sport~"fitness"' in query
    assert "(41.1,-8.7,41.2,-8.5)" in query


def test_addNearestPoi_distance_and_name(pois):
    # 0.001 deg of latitude north of the node: R * pi/180 * 0.001 = 111.19 m
    near_node = (41.151, -8.61)
    near_way = (41.16, -8.6185)
    unnamed = (41.14, -8.60)  # exactly on the unnamed gym
    out = osm.addNearestPoi(stays(near_node, near_way, unnamed), pois, "gym")

    assert out["nearest_gym_m"].iloc[0] == 111
    assert out["nearest_gym_name"].iloc[0] == "Gym Node"
    assert out["nearest_gym_m"].iloc[1] == round(haversineM(*near_way, 41.16, -8.62))
    assert out["nearest_gym_name"].iloc[1] == "Gym Way"
    assert out["nearest_gym_m"].iloc[2] == 0
    assert pd.isna(out["nearest_gym_name"].iloc[2])


def test_addNearestPoi_matches_haversine_within_one_meter(pois):
    stay = (41.155, -8.615)
    out = osm.addNearestPoi(stays(stay), pois, "gym")
    want = min(haversineM(*stay, r.lat, r.lon) for r in pois.itertuples())
    assert abs(out["nearest_gym_m"].iloc[0] - want) <= 1


def test_addNearestPoi_generic_kind_and_empty_pois(pois):
    metro = pois.assign(kind="metro")
    out = osm.addNearestPoi(stays((41.15, -8.61)), metro, "metro")
    assert "nearest_metro_m" in out.columns and "nearest_gym_m" not in out.columns

    empty = osm.addNearestPoi(stays((41.15, -8.61)), pois, "metro")  # no metro pois
    assert pd.isna(empty["nearest_metro_m"].iloc[0])


def test_addNearestPoi_does_not_modify_input(pois):
    original = stays((41.15, -8.61))
    osm.addNearestPoi(original, pois)
    assert list(original.columns) == ["latitude", "longitude"]


def test_addNearestPoi_fills_schema_fields(pois, listings):
    out = osm.addNearestPoi(listings, pois, "gym")
    assert list(out.columns) == list(Listing.model_fields)
    rec = out.iloc[0].to_dict()
    rec = {k: None if not isinstance(v, (list, dict)) and pd.isna(v) else v for k, v in rec.items()}
    assert Listing.model_validate(rec).nearest_gym_m is not None


def test_fetchPois_reuses_saved_file(tmp_path, monkeypatch):
    monkeypatch.setattr(osm, "RAW_DIR", tmp_path)
    saved = tmp_path / "porto" / "osm_gyms_2026-10-22.json"
    saved.parent.mkdir(parents=True)
    saved.write_text((FIXTURES / "overpass_gyms.json").read_text(encoding="utf-8"), encoding="utf-8")

    def noNetwork(*args, **kwargs):
        raise AssertionError("must not call Overpass when the file exists")

    monkeypatch.setattr(osm.httpx, "post", noNetwork)
    raw, path = osm.fetchPois("porto", today=date(2026, 10, 22))
    assert path == saved and len(raw["elements"]) == 4


def test_metroQuery_covers_both_tags_metro_do_porto_may_use():
    query = osm.metroQuery((41.1, -8.7, 41.2, -8.5))
    assert 'station~"subway|light_rail"' in query
    assert "subway=yes" in query and "light_rail=yes" in query
    assert query.count("[!construction]") == 3
    assert "(41.1,-8.7,41.2,-8.5)" in query and "out center" in query


def test_parsePois_metro_station_mapped_twice_keeps_both_and_skips_construction():
    raw = json.loads((FIXTURES / "overpass_metros.json").read_text(encoding="utf-8"))
    stations = osm.parsePois(raw, "metro")
    # way/2004 is a station still being built, next to Ribeira Nomad Loft: not evidence
    assert stations["osm_id"].tolist() == ["node/2001", "relation/2002", "node/2003"]
    assert (stations["kind"] == "metro").all()
    assert stations["name"].nunique() == 2  # the station node and its stop area


def test_fetchPois_asks_overpass_for_the_kind_and_saves_it_apart(tmp_path, monkeypatch):
    monkeypatch.setattr(osm, "RAW_DIR", tmp_path)
    raw = json.loads((FIXTURES / "overpass_metros.json").read_text(encoding="utf-8"))
    sent = []

    def fakePost(url, data, headers, timeout):
        sent.append(data["data"])
        return type("Response", (), {"raise_for_status": lambda self: None, "json": lambda self: raw})()

    monkeypatch.setattr(osm.httpx, "post", fakePost)
    got, path = osm.fetchPois("porto", "metro", today=date(2026, 10, 22))

    assert "light_rail" in sent[0] and "fitness" not in sent[0]
    assert path == tmp_path / "porto" / "osm_metros_2026-10-22.json"
    assert json.loads(path.read_text(encoding="utf-8")) == got == raw


def test_main_adds_gym_and_metro_columns_and_reports_both(processed, tmp_path, monkeypatch):
    out = tmp_path / "processed"
    out.mkdir()
    (out / "listings.parquet").write_bytes((processed / "listings.parquet").read_bytes())
    raw_dir = tmp_path / "raw" / "porto"
    raw_dir.mkdir(parents=True)
    today = date.today().isoformat()  # fetchPois reuses today's files: no network
    for kind in ["gym", "metro"]:
        (raw_dir / f"osm_{kind}s_{today}.json").write_text(
            (FIXTURES / f"overpass_{kind}s.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(osm, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(osm, "processedDir", lambda city: out)

    osm.main(["--city", "porto"])

    listings = readListings(out / "listings.parquet")
    trindade = listings.set_index("listing_id").loc["40001006"]
    assert trindade["nearest_metro_name"] == "Trindade" and trindade["nearest_metro_m"] < 100
    assert listings["nearest_gym_m"].notna().all()
    assert list(listings.columns) == list(Listing.model_fields)
    assert sorted(pd.read_parquet(out / "pois.parquet")["kind"].unique()) == ["gym", "metro"]
    report = json.loads((out / "ingest_report.json").read_text(encoding="utf-8"))["osm"]
    assert (report["gyms"], report["metro_stations"], report["metro_station_names"]) == (3, 3, 2)
    assert report["fetched_date"] == today
