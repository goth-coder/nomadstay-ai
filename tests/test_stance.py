"""
T5.5: precomputed stances (ingest/stance.py), no OpenAI calls.
"""

import json

import pandas as pd

from ingest.stance import batchLines, responseText, reviewGroups, stanceRows


def reviews() -> pd.DataFrame:
    return pd.DataFrame([
        {"listing_id": "L1", "review_id": "r1", "comments": "Fast wifi and a quiet street."},
        {"listing_id": "L1", "review_id": "r2", "comments": "Lovely host, great breakfast."},
        {"listing_id": "L2", "review_id": "r3", "comments": "Metro station next door."},
    ])


def test_review_groups_keep_only_reviews_that_mention_an_aspect():
    groups = reviewGroups(reviews(), size=1)

    assert [[r["review_id"] for r in g] for g in groups] == [["r1"], ["r3"]]
    assert groups[0][0]["aspects"] == ["quiet", "wifi"]
    assert groups[0][0]["id"] == 0  # local id inside the group


def test_batch_lines_are_responses_requests_with_custom_ids():
    lines = [json.loads(line) for line in batchLines(reviewGroups(reviews()), "gpt-4o-mini")]

    assert lines[0]["custom_id"] == "g0"
    assert lines[0]["url"] == "/v1/responses"
    assert lines[0]["body"]["model"] == "gpt-4o-mini"
    assert "Fast wifi" in lines[0]["body"]["input"]


def test_response_text_reads_the_message_output_of_a_batch_body():
    body = {"output": [
        {"type": "reasoning", "content": []},
        {"type": "message", "content": [{"type": "output_text", "text": '{"labels": []}'}]},
    ]}

    assert responseText(body) == '{"labels": []}'


def test_stance_rows_cover_every_pair_and_mark_missing_ones():
    group = reviewGroups(reviews())[0]
    text = group[0]["text"]
    labels = {(0, "wifi"): {"stance": "supports", "strength": 2, "quote": "Fast wifi",
                            "quote_start": text.find("Fast wifi"), "quote_valid": True}}

    rows = stanceRows(group, labels, "gpt-4o-mini", 1.2)

    by_pair = {(r["review_id"], r["aspect"]): r for r in rows}
    assert set(by_pair) == {("r1", "wifi"), ("r1", "quiet"), ("r3", "transport")}
    assert text[by_pair[("r1", "wifi")]["quote_start"]:by_pair[("r1", "wifi")]["quote_end"]] == "Fast wifi"
    assert by_pair[("r1", "quiet")]["stance"] is None  # not labeled → explicit gap
