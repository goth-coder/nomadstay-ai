"""
O que o agente é capaz fazer:
      Capacidade	               Exemplo de pergunta
  Structured filtering	  Find me a place in Porto under €100/night.
  Evidence from reviews	  With reliable Wi-Fi for video calls and quiet at night.
  Map evidence	          With a gym nearby.
  Multi-turn memory	      Tell me more about the second one.
  Open questions	          Is the host responsive? Is it good with kids? (search_reviews)
  Grounding	              Is it available next week? (no: availability is unknown)
"""

import json
import logging
import os
from collections import deque

from dotenv import load_dotenv
from openai import OpenAI

from agent.nomad import NomadStore, get_stay_details, search_reviews, search_stays
from agent.settings import SYSTEM_PROMPT, tools_schema
from observability.llm import tracedCall

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


def dispatcher(tool_name, args, store: NomadStore, client):
    if tool_name == "search_stays":
        return search_stays(**args, store=store, client=client)

    elif tool_name == "get_stay_details":
        return get_stay_details(**args, store=store)

    elif tool_name == "search_reviews":
        return search_reviews(**args, store=store, client=client)

    raise ValueError(f"Unknown tool: {tool_name}")


def runAgent(
    context: deque,
    client: OpenAI,
    store: NomadStore,
    steps_number: int,
    max_output_tokens: int = 1200,
) -> str:

    for _ in range(min(8, (steps_number))):
        # Agent decides what to do
        # Answer without tool call
        # Answer with Call tool(s)
        response = tracedCall(
            client, "agent",
            model="gpt-4o-mini",
            instructions=SYSTEM_PROMPT,
            input=list(context),
            max_output_tokens=max_output_tokens,
            tools=tools_schema,
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]

        if not tool_calls:
            context.append({"role": "assistant", "content": response.output_text})
            return response.output_text

        for item in response.output:
            if item.type == "function_call":
                tool_name = item.name
                tool_args = json.loads(item.arguments)

                logger.info(
                    "tool_call name=%s args=%s call_id=%s",
                    tool_name, tool_args, item.call_id,
                )

                context.append(item.model_dump(exclude_none=True))

                tool_result = dispatcher(tool_name, tool_args, store, client)

                logger.debug("tool_result name=%s result=%s", tool_name, tool_result)

                context.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            if item.type == "message":
                context.append({"role": "assistant", "content": response.output_text})
                break

    # Steps exhausted without a final answer: force one with what was gathered
    logger.warning("Agent reached max steps (%s), forcing final answer", steps_number)
    response = tracedCall(
        client, "agent_final", model="gpt-4o-mini", instructions=SYSTEM_PROMPT, max_output_tokens=max_output_tokens,
        input=list(context), tools=tools_schema, tool_choice="none",
    )
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text


if __name__ == "__main__":
    from agent.nomad import loadStore
    from observability.logging_config import setupLogging

    setupLogging()
    store = loadStore()  # serving set: local copy or S3
    client = OpenAI(api_key=OPENAI_API_KEY)

    context = deque()
    while True:
        userQuery = input("Ask:")

        if userQuery == "q":
            break

        context.append({"role": "user", "content": userQuery})
        answer = runAgent(context, client, store, 5)
        print("Assistant:", answer)
