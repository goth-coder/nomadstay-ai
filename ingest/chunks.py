"""
Split listing texts and guest reviews into chunks for the vector index.

A chunk is at most max_chars long and ends on a sentence boundary when it can.
Listing text (description, neighborhood overview) and review comments stay in
separate chunks, so a listing claim is never mistaken for guest evidence.
Row position == chunk_id == FAISS position.
"""

import re

import pandas as pd
from pydantic import ValidationError

from domain.schemas import Chunk

MAX_CHARS = 600
MIN_CHARS = 15  # shorter pieces carry no usable evidence

# Ends at . ! ? … (or a line break) followed by whitespace
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\s*\n+\s*")

CHUNK_COLUMNS = list(Chunk.model_fields)


def hardSplit(text: str, max_chars: int) -> list[str]:
    # No sentence boundary: cut at the last space before max_chars, else exactly at max_chars
    pieces = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars + 1)
        if cut < max_chars // 2:
            cut = max_chars
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    pieces.append(text)
    return pieces


def splitText(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Windows of at most max_chars on sentence boundaries; pieces under MIN_CHARS are dropped."""
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if len(text) >= MIN_CHARS else []

    sentences = []
    for sentence in SENTENCE_END.split(text):
        sentences += hardSplit(sentence, max_chars) if len(sentence) > max_chars else [sentence]

    windows, current = [], ""
    for sentence in filter(None, (s.strip() for s in sentences)):
        if current and len(current) + 1 + len(sentence) > max_chars:
            windows.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}" if current else sentence
    if current:
        windows.append(current)
    return [w for w in windows if len(w) >= MIN_CHARS]


def textOrEmpty(value) -> str:
    return value if isinstance(value, str) else ""


def buildChunks(listings_df: pd.DataFrame, reviews_df: pd.DataFrame,
                max_chars: int = MAX_CHARS) -> pd.DataFrame:
    """
    Chunks of listing descriptions, neighborhood overviews and review comments,
    with exactly the domain.schemas.Chunk columns. Listing chunks come first,
    then reviews in the order given. Every row is validated with Chunk.
    """
    rows = []
    for listing in listings_df.itertuples():
        for field in ["description", "neighborhood_overview"]:
            for piece in splitText(textOrEmpty(getattr(listing, field)), max_chars):
                rows.append((listing.listing_id, "listing", field, None, None, piece))

    known = set(listings_df["listing_id"])
    for review in reviews_df.itertuples():
        if review.listing_id not in known:
            continue
        for piece in splitText(textOrEmpty(review.comments), max_chars):
            rows.append((review.listing_id, "review", "comments", review.review_id,
                         review.date, piece))

    chunks = pd.DataFrame(rows, columns=CHUNK_COLUMNS[1:])
    chunks.insert(0, "chunk_id", range(len(chunks)))
    return validateChunks(chunks)


def validateChunks(chunks: pd.DataFrame) -> pd.DataFrame:
    # Fail loudly: a bad row would shift or poison the index
    invalid = []
    for record in chunks.to_dict("records"):
        record = {k: None if pd.isna(v) else v for k, v in record.items()}
        try:
            Chunk.model_validate(record)
        except ValidationError as e:
            invalid.append((record["chunk_id"], str(e).splitlines()[1][:120]))
    if invalid:
        raise ValueError(f"{len(invalid)} invalid chunks, e.g. {invalid[:3]}")
    return chunks
