import json
import logging

from dotenv import load_dotenv
from openai import OpenAI

from agent.settings import JUDGE_PROMPT
from legacy.data import HOTELS
from legacy.utils import parseData
from retrieval.rag import VectorStore, similaritySearch

load_dotenv()

logger = logging.getLogger(__name__)


def get_hotel_info(name: str) -> dict:
    if name.lower() not in HOTELS.keys():
        return {"error": "Hotel not in the database"}

    hotel = HOTELS.get(name.lower(), "No hotel found").copy()
    hotel.pop("review")
    return hotel


def filter_hotels(max_price: float | None = None, city: str | None = None):
    filteredHotels = [
        {name: {k: v for k, v in hotel.items() if k != "review"}}
        for name, hotel in HOTELS.items()
        if (max_price is None or hotel["price_per_night"] <= max_price)
        and (city is None or hotel["city"] == city)
    ]
    return filteredHotels


def retriever(
    query: str,
    v_store: VectorStore,
    client: OpenAI,
    top_k: int = 3,
) -> list[str]:
    """
    Gets the documents via similarity search and returns them as strings
    """
    hotels_doc, _ = similaritySearch(query, v_store, top_k)  # Index of the chunks

    retrievedHotels = [parseData(hotel) for hotel in hotels_doc]
    logger.debug(
        "retriever query=%r candidates=%s", query, [list(hotel)[0] for hotel in hotels_doc]
    )

    if retrievedHotels:
        return judge_chunks(retrievedHotels, query, client)
    else:
        return []


def judge_chunks(chunks: list[str], query: str, client: OpenAI) -> list[str]:
    if not chunks:
        return []

    judge_input = f"Query: {query}\n\nCandidates ({len(chunks)}):\n\n" + "\n\n".join(
        f"[{i}] {chunk}" for i, chunk in enumerate(chunks)
    )
    try:
        answer = client.responses.create(
            model="gpt-4o-mini",
            instructions=JUDGE_PROMPT,
            input=judge_input,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "relevance",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "relevant": {"type": "array", "items": {"type": "boolean"}}
                        },
                        "required": ["relevant"],
                        "additionalProperties": False,
                    },
                }
            },
        )
        flags = json.loads(answer.output_text)["relevant"]

        if len(flags) != len(chunks):
            raise ValueError("Judge returned an unexpected number of flags")

        logger.debug("judge flags=%s", flags)
        return [chunk for chunk, keep in zip(chunks, flags) if keep]

    except Exception:
        logger.warning(
            "Judge failed, returning all %s candidates", len(chunks), exc_info=True
        )
        return chunks
