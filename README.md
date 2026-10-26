# NomadStay AI

An LLM agent that helps digital nomads choose an Airbnb stay. You give a city,
a nightly budget and what you need (fast Wi-Fi, a desk, quiet nights, a gym
nearby). It ranks the stays and **keeps what the host claims apart from what
guests actually reported**, quoting the guests.

Data: [Inside Airbnb](https://insideairbnb.com) Porto snapshot of 2026-09-19
(11,190 listings, 1.18M reviews) and gyms from OpenStreetMap.
Not affiliated with Airbnb.

```
POST /agent/run  {"message": "Un logement calme à Porto pour moins de 120 € la nuit,
                              avec une salle de sport à proximité."}

2. Vintage Porto
   - Price: €104.00/night, indicative on 2026-09-19 (not a live quote)
   - Gym nearby: private gym in the building, confirmed by guests.
     "Gym was very small but a nice bonus." (May 31, 2026)
   - Quiet: reported by guests, not stated by the host.
     "The building was safe and quiet." (July 11, 2026)
3. Ceuta Inn Porto
   - Quiet: some guests reported noise.
     "Mainly because the building is very noisy." (July 29, 2023)
```

## How it works

```
offline (once per snapshot)                        online (per question)
──────────────────────────                         ─────────────────────
Inside Airbnb CSVs ─► clean, filter, sample        user message
OpenStreetMap gyms ─► nearest gym per listing          │
reviews ─► chunks ─► embeddings ─► FAISS           gpt-4o-mini agent loop (tool calling)
review × aspect ─► LLM judge (Batch API)               │
   supports / contradicts + quote,                     ├─ search_stays: filters in code (city,
   quote checked in code                               │   price, must-haves) ─► evidence ─► ranking
          │                                            ├─ get_stay_details
          ▼                                            └─ search_reviews: open questions
versioned serving set on S3 ─────────────────────►         (BM25 + embeddings, judged at query time)
(manifest with file hashes)                            │
                                                   answer ─► citation check (quotes must exist
                                                             in the tool output)
```

- **Filters run in code, never in the LLM.** City, price and must-have
  amenities are applied by `search_stays`; the model only picks the arguments.
- **Three kinds of evidence, kept apart:** the host's claim (amenities list),
  guest reviews (quote + date) and the map (gym distance). Each aspect gets a
  verdict: `verified` (claimed and confirmed), `claim_only`, `evidence_only`,
  `mixed`, `contradicted`, `unknown`. Ranking weighs recent reviews more and
  penalises contradictions.
- **Precomputed stance.** gpt-4o-mini labelled 60,296 review × aspect pairs
  through the OpenAI Batch API (about $1.42). At query time the agent reads
  these labels; it only calls the judge for paraphrases the keywords missed and
  for open questions.
- **Grounding checks in code.** A judge's quote is kept only if it is found in
  the review. Every quote in the final answer is checked against the tool
  outputs; anything else is returned in `warnings.unverified_quotes`.
  Reviews are treated as untrusted data (prompt injection).
- **No agent framework.** Plain Python functions and one loop in `agent/agent.py`.

## Results

Gold labels were made with Claude Opus, following the same written guidelines
as the judge (`domain/stance_guidelines.yaml`). They are used only to measure,
never served.

**Retrieval** (finding the review chunks that talk about an aspect).
Exhaustive gold set of 2,169 labels, `eval/run_retrieval_eval.py`:

| Method                                      | Recall@4       | nDCG@10        |
| ------------------------------------------- | -------------- | -------------- |
| Keywords + multilingual embeddings (served) | **0.76** | **0.84** |
| Keywords only                               | 0.74           | 0.81           |
| BM25                                        | 0.54           | 0.62           |
| Multilingual embeddings only                | 0.49           | 0.60           |

**Stance judge** (supports / contradicts / irrelevant). Held-out half
(148 labels) of a 300-label gold set, `eval/run_stance_eval.py`:
macro-F1 **0.88**, contradiction recall **0.93**, accuracy 0.91.

**Agent, end to end.** 25 scenarios (`eval/queries.yaml`: 4 languages, refusals
for other cities and currencies, availability and pet questions),
`eval/run_agent_eval.py`, same checks before and after the T8 fixes:

| Check                                                 | Before | After       |
| ----------------------------------------------------- | ------ | ----------- |
| Answers with a price but no snapshot date             | 15     | **0** |
| Searches with needs the user never asked for          | 10     | **2** |
| Wrong status (e.g. a guessed currency)                | 1      | **0** |
| Wi-Fi speed called "verified" (it is the host's test) | 9      | 6           |
| Answers with an edited or translated quote            | 4      | 5           |

The last two are the known limits of gpt-4o-mini here; the citation check
catches the edited quotes at runtime. A full run costs about $0.025.

## Observability

JSON logs (`observability/`), one line per event, all carrying the request id
(also returned in the `X-Request-ID` header):

- `llm_call`: purpose, model, latency, input/cached/output tokens, cost in USD.
- `request`: route template, status, duration, model calls, tokens and cost of
  the whole request (RED metrics). Percentiles come from CloudWatch Logs Insights:

```
filter message = "request" and route = "/agent/run"
| stats pct(duration_ms, 50), pct(duration_ms, 95), avg(cost_usd) by bin(5m)
```

## Run it

Python 3.12. The serving set (`data/refined/v3/porto`) comes from the
offline pipeline below or from your S3 bucket.

```bash
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-dev.txt

# .env: OPENAI_API_KEY, API_KEY (clients send it as X-API-Key),
#       BUCKET and AWS_REGION (only to download the serving set), LOG_LEVEL
uv run python -m uvicorn api.app:app --reload
```

```bash
curl -s localhost:8000/agent/run -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"message": "Porto under 90 euros, quiet, I have video calls all day"}'
# follow-up: pass the returned session_id
# open question: "Is the host responsive?" (search_reviews)
```

Offline pipeline (once per snapshot):

```bash
uv run python -m ingest.insideairbnb --city porto      # download, clean, sample reviews
uv run python -m ingest.osm --city porto               # nearest gym per listing
uv run python -m ingest.build --city porto --model multilingual-minilm --version 2
uv run python -m ingest.stance submit --city porto     # Batch API; then: collect
uv run python -m ingest.package --city porto --from-version 2 --version 3 [--upload]
```

Tests (no OpenAI or S3 calls; a synthetic mini snapshot in `tests/fixtures/`):

```bash
uv run python -m pytest -q
```

## Deploy

Docker image on AWS ECS Fargate (1 vCPU, 3 GB), secrets from Secrets Manager,
serving set downloaded from S3 at startup, health check on `/health`.
Task definition and IAM policies with placeholders: `deploy/*.example.json`.

## Layout

```
api/            FastAPI app, API key auth, sessions
agent/          agent loop, tools (nomad.py), prompts and tool schemas, judge
domain/         pure logic: schemas, aspects and keywords, ranking, judge guidelines
retrieval/      catalog filters, evidence store, FAISS index, embeddings, BM25
ingest/         offline CLIs: insideairbnb → osm → build → stance → package
eval/           gold sets, metrics and the three evaluations
observability/  JSON logs, request id, LLM call tracing, request log
storage/        S3 artifacts
```

## Limits

- One city (Porto) and one snapshot: prices are indicative, availability unknown.
- Gym distances are straight lines; listing locations are shifted up to ~150 m.
- Sessions live in memory: one task, no horizontal scaling yet.
- gpt-4o-mini still edits a quote or overstates "verified" now and then (see Results).

## Data and license

Code: MIT. Listings and reviews: Inside Airbnb, CC BY 4.0 (also the review
excerpts in `eval/labels/`). Gyms: © OpenStreetMap contributors, ODbL.
