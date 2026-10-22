import json
from types import SimpleNamespace

import pytest

from retrieval import rag


class FakeResponses:
    """
    Stands in for client.responses: returns the queued responses in order
    and records every call, so tests never hit the OpenAI API.
    """

    def __init__(self, outputs: list):
        self.outputs = list(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def fakeClient(*outputs) -> SimpleNamespace:
    return SimpleNamespace(responses=FakeResponses(outputs))


def textResponse(text: str) -> SimpleNamespace:
    # Final answer: a single "message" item, no tool calls
    return SimpleNamespace(output_text=text, output=[SimpleNamespace(type="message")])


def judgeResponse(flags: list[bool]) -> SimpleNamespace:
    return SimpleNamespace(output_text=json.dumps({"relevant": flags}), output=[])


class FakeFunctionCall(SimpleNamespace):
    def model_dump(self, exclude_none=True):
        return {
            "type": "function_call",
            "name": self.name,
            "arguments": self.arguments,
            "call_id": self.call_id,
        }


def toolCallResponse(name: str, args: dict, call_id: str = "call_1") -> SimpleNamespace:
    item = FakeFunctionCall(
        type="function_call", name=name, arguments=json.dumps(args), call_id=call_id
    )
    return SimpleNamespace(output_text="", output=[item])


@pytest.fixture(scope="session")
def v1_store(tmp_path_factory) -> rag.VectorStore:
    """
    Builds the v1 index (data.HOTELS) in a temp dir,
    without touching data/refined or S3.
    """
    artifact_dir = tmp_path_factory.mktemp("artifacts")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(rag, "ARTIFACT_DIR", artifact_dir)
        mp.setattr(rag, "INDEX_PATH", artifact_dir / "hotels.index")
        mp.setattr(rag, "DOCS_PATH", artifact_dir / "hotels_doc.json")
        index, docs = rag.buildIndex()
    return rag.VectorStore(index, docs)
