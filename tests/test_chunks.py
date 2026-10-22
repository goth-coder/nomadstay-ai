import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from domain.schemas import Chunk
from ingest.chunks import (
    CHUNK_COLUMNS,
    MIN_CHARS,
    buildChunks,
    splitText,
    validateChunks,
)
from ingest.insideairbnb import normalizeListings, normalizeReviews, readListingsCsv

MINI = Path(__file__).parent / "fixtures" / "mini_porto"
SNAPSHOT = json.loads((MINI / "expected.json").read_text(encoding="utf-8"))["snapshot_date"]


@pytest.fixture(scope="module")
def mini():
    listings = normalizeListings(readListingsCsv(MINI / "listings.csv"), "porto", SNAPSHOT)
    raw = pd.read_csv(MINI / "reviews.csv", dtype=str, keep_default_na=False)
    return listings, normalizeReviews(raw, set(listings["listing_id"]))


def listingsFrame(*rows):
    return pd.DataFrame(rows, columns=["listing_id", "description", "neighborhood_overview"])


def reviewsFrame(*rows):
    return pd.DataFrame(rows, columns=["listing_id", "review_id", "date", "comments"])


# ---------- splitText ----------

def test_short_text_is_one_chunk():
    assert splitText("A short but valid sentence.") == ["A short but valid sentence."]


def test_pieces_under_min_chars_are_dropped():
    assert splitText("Too short") == []
    assert splitText("") == [] and splitText(None) == []
    assert len("Too short") < MIN_CHARS


def test_long_text_splits_on_sentence_boundaries_within_max_chars():
    sentences = [f"Sentence number {i} is about the apartment." for i in range(30)]
    pieces = splitText(" ".join(sentences), max_chars=200)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)
    assert all(p.endswith(".") for p in pieces)  # never cut mid-sentence
    assert " ".join(pieces) == " ".join(sentences)  # nothing lost


def test_newlines_are_sentence_boundaries():
    text = "First paragraph without final punctuation\n\nSecond paragraph that is also long"
    assert splitText(text, max_chars=50) == [
        "First paragraph without final punctuation",
        "Second paragraph that is also long",
    ]


def test_sentence_longer_than_max_is_hard_split_on_spaces():
    text = " ".join(["word"] * 100)  # 499 chars, no sentence end
    pieces = splitText(text, max_chars=100)
    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces) == text


def test_hard_split_without_any_space():
    pieces = splitText("x" * 250, max_chars=100)
    assert [len(p) for p in pieces] == [100, 100, 50]


# ---------- buildChunks ----------

def test_columns_ids_and_sources():
    listings = listingsFrame(
        ("1", "A bright apartment near the river.", "Cafes and shops all around."),
        ("2", "Second listing description here.", None),
    )
    reviews = reviewsFrame(
        ("1", "r1", date(2026, 1, 5), "Great stay, very clean and quiet."),
        ("2", "r2", date(2026, 2, 6), "Loved it, would book again."),
    )
    chunks = buildChunks(listings, reviews)

    assert list(chunks.columns) == CHUNK_COLUMNS
    assert chunks["chunk_id"].tolist() == list(range(len(chunks)))
    assert chunks[["listing_id", "source", "field"]].values.tolist() == [
        ["1", "listing", "description"],
        ["1", "listing", "neighborhood_overview"],
        ["2", "listing", "description"],
        ["1", "review", "comments"],
        ["2", "review", "comments"],
    ]


def test_review_chunks_carry_review_id_and_date_listing_chunks_do_not():
    listings = listingsFrame(("1", "A bright apartment near the river.", None))
    reviews = reviewsFrame(("1", "r1", date(2026, 1, 5), "Great stay, very clean and quiet."))
    chunks = buildChunks(listings, reviews)

    review = chunks[chunks["source"] == "review"].iloc[0]
    assert (review["review_id"], review["date"]) == ("r1", date(2026, 1, 5))
    listing = chunks[chunks["source"] == "listing"].iloc[0]
    assert pd.isna(listing["review_id"]) and pd.isna(listing["date"])


def test_long_review_splits_and_every_piece_keeps_its_review_id():
    comment = " ".join(f"Sentence {i} of a very long review." for i in range(40))
    chunks = buildChunks(
        listingsFrame(("1", "A bright apartment near the river.", None)),
        reviewsFrame(("1", "r1", date(2026, 1, 5), comment)),
        max_chars=200,
    )
    pieces = chunks[chunks["source"] == "review"]
    assert len(pieces) > 1 and (pieces["review_id"] == "r1").all()
    assert pieces["text"].str.len().max() <= 200


def test_reviews_of_unknown_listings_are_skipped():
    chunks = buildChunks(
        listingsFrame(("1", "A bright apartment near the river.", None)),
        reviewsFrame(("999", "r1", date(2026, 1, 5), "Review of a listing that was filtered out.")),
    )
    assert chunks["source"].tolist() == ["listing"]


def test_missing_neighborhood_overview_is_fine():
    chunks = buildChunks(listingsFrame(("1", "A bright apartment near the river.", float("nan"))),
                         reviewsFrame())
    assert chunks["field"].tolist() == ["description"]


def test_fixture_chunks_validate_and_keep_languages(mini):
    listings, reviews = mini
    chunks = buildChunks(listings, reviews)
    for record in chunks.to_dict("records"):
        record = {k: None if pd.isna(v) else v for k, v in record.items()}
        Chunk.model_validate(record)
    assert chunks["chunk_id"].tolist() == list(range(len(chunks)))
    texts = chunks["text"]
    assert texts.str.startswith("Le wifi était très rapide").any()
    assert texts.str.startswith("Há um ginásio mesmo ao lado").any()
    assert not chunks["text"].str.contains("<", regex=False).any()


def test_validateChunks_fails_loudly_on_invalid_row():
    bad = pd.DataFrame([{
        "chunk_id": 0, "listing_id": "1", "source": "review", "field": "comments",
        "review_id": None, "date": None, "text": "A review without an id.",
    }])
    with pytest.raises(ValueError, match="invalid chunks"):
        validateChunks(bad)
