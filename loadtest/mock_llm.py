"""
Mock OpenAI client for load tests: same shape as the real one, waits like
the real one, costs nothing. Enabled with MOCK_LLM=1 (never in production).

Each agent turn takes two model calls, as with the real model: the first asks
for search_stays, the second writes the answer. Judge calls (paraphrases,
search_reviews) label every pair "irrelevant", so only the precomputed evidence
is used, with as many calls as a real judge would take.
"""

import json
import random
import re
import time
import uuid
from types import SimpleNamespace

# Varied searches, so the load exercises different filters and rankings
SEARCHES = [
    {"city": "Porto", "max_price": 100, "currency": "EUR", "must_have": [],
     "preferences": ["wifi", "quiet"], "max_gym_distance_m": None},
    {"city": "Porto", "max_price": 80, "currency": "EUR", "must_have": ["workspace"],
     "preferences": ["workspace"], "max_gym_distance_m": None},
    {"city": "Porto", "max_price": 120, "currency": "EUR", "must_have": [],
     "preferences": ["quiet", "gym_nearby"], "max_gym_distance_m": None},
    {"city": "Porto", "max_price": None, "currency": None, "must_have": [],
     "preferences": ["wifi", "workspace", "quiet"], "max_gym_distance_m": None},
    {"city": "Porto", "max_price": 150, "currency": "EUR", "must_have": [],
     "preferences": ["transport"], "max_gym_distance_m": None},
]


REVIEW = re.compile(r'<review id="(\d+)" aspects="([^"]*)">')
EXCERPT = re.compile(r'<excerpt id="(\d+)">')


def judgeLabels(judge_input: str) -> str:
    """'irrelevant' for every pair asked, like a real judge that found nothing.
    No labels at all would read as skipped pairs and trigger a retry call."""
    labels = [{"id": int(i), "aspect": aspect.strip(), "quote": "", "stance": "irrelevant", "strength": 0}
              for i, aspects in REVIEW.findall(judge_input) for aspect in aspects.split(",")]
    labels += [{"id": int(i), "quote": "", "stance": "irrelevant", "strength": 0}
               for i in EXCERPT.findall(judge_input)]
    return json.dumps({"labels": labels})


class MockFunctionCall(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {"type": "function_call", "name": self.name,
                "arguments": self.arguments, "call_id": self.call_id}


def toolAnsweredThisTurn(items: list) -> bool:
    """A function_call_output after the last user message: time to write the answer."""
    last_user = max((i for i, item in enumerate(items) if item.get("role") == "user"), default=-1)
    return any(item.get("type") == "function_call_output" for item in items[last_user + 1:])


class MockResponses:
    def __init__(self, min_s: float, max_s: float):
        self.min_s, self.max_s = min_s, max_s

    def create(self, **params):
        # sleep uses no CPU, like waiting for OpenAI: it holds a server thread, not a core
        time.sleep(random.uniform(self.min_s, self.max_s))

        if "tools" not in params:  # the judge
            return SimpleNamespace(output=[], output_text=judgeLabels(params["input"]), usage=None)

        if toolAnsweredThisTurn(params["input"]):
            return SimpleNamespace(output=[SimpleNamespace(type="message")],
                                   output_text="Here are the best stays I found (mock answer).",
                                   usage=None)

        call = MockFunctionCall(type="function_call", name="search_stays",
                                arguments=json.dumps(random.choice(SEARCHES)),
                                call_id=f"call_{uuid.uuid4().hex[:12]}")
        # usage=None: zero tokens, so mock calls never count as real cost
        return SimpleNamespace(output=[call], output_text="", usage=None)


class MockOpenAI:
    def __init__(self, min_s: float = 1.0, max_s: float = 2.0):
        self.responses = MockResponses(min_s, max_s)

    def close(self):
        pass
