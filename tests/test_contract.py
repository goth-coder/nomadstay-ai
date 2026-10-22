"""
T1: data contract (schemas.py), aspect detection (aspects.py)
and integrity of the mini_porto fixture.
"""

import csv
import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from domain.aspects import ASPECTS, claimedAspects, mentionedAspects, parseWifiMbps
from domain.schemas import Chunk, Listing

FIXTURE = Path(__file__).parent / "fixtures" / "mini_porto"


def readCsv(name: str) -> list[dict]:
    with open(FIXTURE / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads((FIXTURE / "expected.json").read_text(encoding="utf-8"))["listings"]


# schemas


def listing(**overrides) -> dict:
    base = {
        "listing_id": "1436961951661730861", "name": "Loft", "url": "https://www.airbnb.com/rooms/1",
        "city": "porto", "latitude": 41.14, "longitude": -8.62, "price": 85.0, "currency": "EUR",
        "price_status": "listed", "snapshot_date": date(2026, 9, 19),
    }
    return base | overrides


def test_listing_accepts_a_priced_listing():
    assert Listing(**listing()).price == 85.0


def test_listing_accepts_missing_price_only_with_missing_status():
    assert Listing(**listing(price=None, price_status="missing")).price is None

    with pytest.raises(ValidationError):
        Listing(**listing(price=None, price_status="listed"))
    with pytest.raises(ValidationError):
        Listing(**listing(price=85.0, price_status="missing"))


def test_listing_rejects_out_of_range_values():
    with pytest.raises(ValidationError):
        Listing(**listing(price=0))
    with pytest.raises(ValidationError):
        Listing(**listing(rating=95))  # 0-5 scale, not 0-100


def test_chunk_review_needs_review_id_and_comments_field():
    review = Chunk(chunk_id=0, listing_id="1", source="review", field="comments",
                   review_id="7", date=date(2026, 9, 1), text="Fast wifi")
    assert review.review_id == "7"

    with pytest.raises(ValidationError):
        Chunk(chunk_id=0, listing_id="1", source="review", field="comments", text="x")
    with pytest.raises(ValidationError):
        Chunk(chunk_id=0, listing_id="1", source="listing", field="comments", text="x")
    with pytest.raises(ValidationError):
        Chunk(chunk_id=0, listing_id="1", source="listing", field="description",
              review_id="7", text="x")


def test_chunk_rejects_empty_text():
    with pytest.raises(ValidationError):
        Chunk(chunk_id=0, listing_id="1", source="listing", field="description", text="")


# aspects: claims come only from the amenity list


@pytest.mark.parametrize(
    "amenity, aspect",
    [
        ("Wifi", "wifi"),
        ("Fast wifi – 300 Mbps", "wifi"),
        ("Wifi – 500 Mbps", "wifi"),
        ("Pocket wifi", "wifi"),
        ("Ethernet connection", "wifi"),
        ("Dedicated workspace", "workspace"),
        ("Laptop friendly workspace", "workspace"),
        ("Gym", "gym_nearby"),
        ("Shared gym in building", "gym_nearby"),
        ("Private gym in building", "gym_nearby"),
        ("Shared gym nearby", "gym_nearby"),
    ],
)
def test_claimed_aspects_recognizes_real_amenity_names(amenity, aspect):
    assert claimedAspects(["Kitchen", amenity]) == {aspect: amenity}


@pytest.mark.parametrize(
    "amenity",
    ["Sound system", "Bluetooth sound system", "Noise decibel monitors on property",
     "Kitchen", "TV with standard cable"],
)
def test_claimed_aspects_ignores_lookalikes(amenity):
    assert claimedAspects([amenity]) == {}


def test_quiet_and_transport_can_never_be_claimed():
    assert ASPECTS["quiet"].amenity is None
    assert ASPECTS["transport"].amenity is None


def test_parse_wifi_mbps_takes_the_highest_listed_speed():
    assert parseWifiMbps(["Fast wifi – 199 Mbps", "Wifi – 500 Mbps"]) == 500
    assert parseWifiMbps(["Fast wifi - 50 Mbps"]) == 50
    assert parseWifiMbps(["Wifi", "Ethernet connection"]) is None


# aspects: mentions in free text (multilingual)


@pytest.mark.parametrize(
    "text, aspects",
    [
        ("The wifi dropped during my Zoom calls", {"wifi", "workspace"}),
        ("Le wifi était très rapide", {"wifi"}),
        ("El wifi no funcionaba", {"wifi"}),
        ("Internet excelente", {"wifi"}),
        ("Apartamento muito silencioso e sossegado", {"quiet"}),
        ("Appartement très calme, aucun bruit", {"quiet"}),
        ("Muy ruidoso por la noche", {"quiet"}),
        ("Há um ginásio mesmo ao lado", {"gym_nearby"}),
        ("There is a gym around the corner", {"gym_nearby"}),
        ("Trindade metro station is 2 minutes away", {"transport"}),
        ("Great location and a very friendly host.", set()),
    ],
)
def test_mentioned_aspects(text, aspects):
    assert mentionedAspects(text) == aspects


@pytest.mark.parametrize("text", ["business trip", "PlayStation", "training", "internetcafe"])
def test_mentioned_aspects_matches_whole_words_only(text):
    assert mentionedAspects(text) == set()


# mini_porto fixture


REAL_LISTING_COLUMNS = {  # subset of the real Porto 2026-09-19 header
    "id", "listing_url", "last_scraped", "name", "description", "neighborhood_overview",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude",
    "property_type", "room_type", "accommodates", "amenities", "price",
    "price_quote_price_per_night", "price_quote_raw", "number_of_reviews", "last_review",
    "review_scores_rating",
}


def test_fixture_uses_real_inside_airbnb_columns():
    assert set(readCsv("listings.csv")[0]) == REAL_LISTING_COLUMNS
    assert set(readCsv("reviews.csv")[0]) == {
        "listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments",
    }


def test_fixture_reviews_reference_existing_listings():
    listing_ids = {row["id"] for row in readCsv("listings.csv")}
    reviews = readCsv("reviews.csv")

    assert {row["listing_id"] for row in reviews} <= listing_ids
    assert len({row["id"] for row in reviews}) == len(reviews)


def test_fixture_claims_match_expected(expected):
    for row in readCsv("listings.csv"):
        amenities = json.loads(row["amenities"])
        labels = expected[row["id"]]

        assert set(claimedAspects(amenities)) == set(labels["claims"]), row["name"]
        assert parseWifiMbps(amenities) == labels.get("wifi_mbps"), row["name"]


def test_fixture_scenario_reviews_mention_their_aspects(expected):
    by_listing: dict[str, set] = {}
    for row in readCsv("reviews.csv"):
        by_listing.setdefault(row["listing_id"], set()).update(mentionedAspects(row["comments"]))

    for listing_id, labels in expected.items():
        assert set(labels["stance"]) <= by_listing.get(listing_id, set()), labels["name"]


def test_fixture_covers_the_hard_cases(expected):
    listings = readCsv("listings.csv")
    reviews = readCsv("reviews.csv")

    assert any("," in row["price"] for row in listings)  # "$1,050.00"
    assert any(row["price"] == "" and row["price_quote_price_per_night"] for row in listings)
    assert any(row["price"] == "" and not row["price_quote_price_per_night"] for row in listings)
    assert any(len(row["id"]) == 19 for row in listings)
    assert any("<br" in row["description"] for row in listings)
    assert any("<br" in row["comments"] for row in reviews)
    assert any(not labels["in_porto"] for labels in expected.values())
    assert any(labels.get("injection") for labels in expected.values())
    assert any(labels.get("stale") for labels in expected.values())
