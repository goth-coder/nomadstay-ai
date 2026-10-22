"""
O que o agente é capaz fazer:
      Capacidade	               Exemplo de pergunta
  Semantic retrieval	    Which hotels have noise complaints?
  Structured filtering	  Find hotels under €100.
  Multi-turn memory	      Which of those has better Wi-Fi?
  Comparative reasoning	  Is it quieter than Aurora Suites?
  Preference tracking	    I need reliable Wi-Fi for remote work.
  Grounding	Does Nordic   Loft offer airport transfers?
"""

import json
import logging
import os
from collections import deque
from pprint import pprint

from dotenv import load_dotenv
from openai import OpenAI

from agent.settings import SYSTEM_PROMPT, tools_schema
from agent.tools import filter_hotels, get_hotel_info, retriever
from retrieval.rag import VectorStore

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

logger = logging.getLogger(__name__)


def dispatcher(tool_name, args, v_store, client):
    if tool_name == "get_hotel_info":
        return get_hotel_info(**args)

    elif tool_name == "filter_hotels":
        return filter_hotels(**args)

    elif tool_name == "retriever":
        return retriever(**args, v_store=v_store, client=client)

    raise ValueError(f"Unknown tool: {tool_name}")


def runAgent(
    context: str,
    client: OpenAI,
    v_store: VectorStore,
    steps_number: int,
) -> str:

    for _ in range(min(8, (steps_number))):
        # Agent decides what to do
        # Answer without tool call
        # Answer with Call tool(s)
        # Answer with Run RAG system
        response = client.responses.create(
            model="gpt-4o-mini",
            instructions=SYSTEM_PROMPT,
            input=list(context),
            max_output_tokens=500,
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

                tool_result = dispatcher(tool_name, tool_args, v_store, client)

                logger.debug("tool_result name=%s result=%s", tool_name, tool_result)

                context.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(tool_result),
                    }
                )
            if item.type == "message":
                context.append({"role": "assistant", "content": response.output_text})
                break

    # Steps exhausted without a final answer: force one with what was gathered
    logger.warning("Agent reached max steps (%s), forcing final answer", steps_number)
    response = client.responses.create(
        model="gpt-4o-mini", instructions=SYSTEM_PROMPT,
        input=list(context), tools=tools_schema, tool_choice="none",
    )
    context.append({"role": "assistant", "content": response.output_text})
    return response.output_text



if __name__ == "__main__":
    from logging_config import setupLogging
    from retrieval.rag import loadOrBuildIndex

    setupLogging()
    v_store = loadOrBuildIndex()  # FAISS vector store: index + hotels doc
    client = OpenAI(api_key=OPENAI_API_KEY)

    context = deque()
    while True:
        userQuery = input("Ask:")

        if userQuery == "q":
            break

        context.append({"role": "user", "content": userQuery})

        args = [context, client, v_store, 5]
        answer = runAgent(*args)
        pprint(context, indent=2)

        print("Assistant:", answer)
