# Agentic RAG Router: Sub-query Division

Week 3, Assignment 2 for the Maven AI course. The notebook ([`001_Agentic_Router.ipynb`](001_Agentic_Router.ipynb)) builds an **agentic RAG** system in which an LLM router decides *where* to look for an answer before any retrieval happens. **Part 1** extends the system to handle compound questions: the query is split into sub-queries, each sub-query is routed independently, and the answers are combined into one response.

---

## 1. How the system works

In a standard RAG pipeline, every question goes through the same retriever. In this system, a router first classifies the query and then dispatches it to the matching tool:

```
User query → Router (LLM classifier) → pick a source → retrieve / search → grounded answer with citations
```

| Route label | Source | Backend |
|---|---|---|
| `OPENAI_QUERY` | OpenAI documentation (Agents) | Qdrant collection `opnai_data` |
| `10K_DOCUMENT_QUERY` | Uber 2021 & Lyft 2024 SEC 10-K filings | Qdrant collection `10k_data` |
| `INTERNET_QUERY` | Everything else (news, trends, cross-vendor comparisons) | SerpApi (Google Search API) |

### Key concepts

- **Router.** `route_query()` uses a system prompt with few-shot examples. The LLM returns JSON of the form `{action, reason, answer}`. The routing decision is itself an LLM call, so it adds cost and latency.
- **Embeddings.** Text is embedded locally with `nomic-embed-text-v1.5` (mean pooling over token vectors). This step does not use the OpenAI API.
- **Semantic retrieval.** `client.query_points(collection, query=vector, limit=3)` returns the three most similar chunks.
- **Grounded generation.** `rag_formatted_response()` tells the model to answer only from the retrieved context and to cite sources as `[1][2]`.
- **RBAC (Section 6).** A permission check runs after the router decides and before any tool is called.
  - Permissions are an allow-list, so anything not explicitly allowed is denied.
  - The order is: identity → route → permission → retrieval.
  - Semantic caching combined with RBAC can leak data. If the cache is keyed only on the question, a finance answer cached for one role can be served to a role that should not see it.

---

## 2. Part 1 requirements

**Goal:** split compound queries into focused sub-queries, run each one through the full agentic pipeline, and compose a single coherent answer.

| # | Requirement | Where it is implemented |
|---|---|---|
| 1 | Split the query with `sub_queries()` | Step 1 of `agentic_rag_multi()` |
| 2 | Route each sub-query separately (sources may differ) | `answer_sub_query()` |
| 3 | Combine the sub-answers into **one** final response | `compose_answer()` |
| 4 | Preserve citations from every sub-answer | `renumber_citations()` |
| 5 | No regression on single questions (no LLM calls beyond the split) | Early return when `len(parts) == 1` |
| 6 | Parse the model's JSON defensively and fall back to one query on failure | `parse_sub_queries()` |

**Self-check table**

| Query | Expected behaviour |
|---|---|
| `what was uber revenue in 2021?` | 1 sub-query, 1 route |
| `what was lyft revenue in 2021 and what was uber revenue in 2021` | 2 sub-queries, both `10K_DOCUMENT_QUERY` |
| `what was uber's 2021 revenue and what are the newest LLMs?` | 2 sub-queries, **different** routes |

---

## 3. Key code

The full implementation is in the Part 1 cell of the notebook.

### ① Defensive parsing

```python
def parse_sub_queries(raw: str, user_query: str) -> list:
    try:
        match = re.search(r"\{.*\}", raw, re.DOTALL)   # grab {...}; ignores surrounding prose and ```json fences
        subs = json.loads(match.group())["subQuestions"]
        subs = [s.strip() for s in subs if isinstance(s, str) and s.strip()]
        return subs or [user_query]                    # an empty list also falls back
    except Exception:
        return [user_query]                            # any failure: treat the input as one query
```

`sub_queries()` returns a **string**, not a dict. The model may wrap its JSON in prose or a code fence, so calling `json.loads` on the raw output would fail.

### ② Route each sub-query independently

```python
def answer_sub_query(sub_query: str) -> dict:
    decision = route_query(sub_query)
    action = decision.get("action")
    route_function = routes.get(action)
    if action in ["OPENAI_QUERY", "10K_DOCUMENT_QUERY"]:
        answer = asyncio.run(route_function(sub_query, action))   # retrieval is async
    else:
        answer = route_function(sub_query, action)                # web search is sync
    return {"query": sub_query, "action": action, "answer": answer, ...}
```

This reuses the existing `routes` table and the same dispatch logic as `agentic_rag()`. Calling `asyncio.run` inside Jupyter works because of the earlier `nest_asyncio.apply()`.

### ③ Renumber citations so they don't collide

```python
def renumber_citations(text: str, offset: int):
    nums = [int(n) for n in re.findall(r"\[(\d+)\]", text)]
    shifted = re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1)) + offset}]", text)
    return shifted, offset + (max(nums) if nums else 0)
```

Each sub-answer numbers its citations from `[1]`, so a plain merge would produce duplicate labels. The code shifts the numbers before synthesis, and the compose prompt tells the model to keep them unchanged. Renumbering in code is more reliable than asking the LLM to do it.

### ④ Single question: no extra LLM call

```python
result = parts[0]["answer"] if len(parts) == 1 else compose_answer(user_query, parts)
```

---

## 4. Results

```
👤 what was uber revenue in 2021?
🔀 1 sub-query → 📍 10K_DOCUMENT_QUERY
🤖 Uber's revenue in 2021 was $17.455 billion.[1]

👤 what was lyft revenue in 2021 and what was uber revenue in 2021
🔀 2 sub-queries → 📍 10K_DOCUMENT_QUERY ×2
🤖 - Lyft revenue: $3.208 billion [1]
   - Uber revenue: $17.455 billion [2]

👤 what was uber's 2021 revenue and what are the newest LLMs?
🔀 2 sub-queries → 📍 10K_DOCUMENT_QUERY + 📍 INTERNET_QUERY
🤖 Uber's 2021 revenue was $17.455 billion. [1]
   The provided information does not identify any current or newest LLMs.
   The listed results [2]–[6] concern ... not LLMs.
```

All three self-checks pass. In the third query, the web results were renumbered to `[2]–[6]`, which shows the citation offset working.

**Observation:** SerpApi returned noisy Google results for the LLM question. This is a quality issue with the upstream search tool, not with the split or routing logic. The composer stated that no relevant information was found instead of inventing an answer, which is the expected grounded behaviour.

---

## 5. Running it locally

1. Create a Python 3.11 virtual environment. `transformers==4.48.0` has no prebuilt wheels for Python 3.14.
   ```bash
   python3.11 -m venv .venv
   .venv/bin/pip install openai qdrant_client "transformers==4.48.0" torch einops python-dotenv nest_asyncio requests matplotlib ipykernel
   ```
2. Create a `.env` file next to the notebook:
   ```
   OPENAI_API_KEY=...
   SERP_API_KEY=...
   ```
3. The prebuilt Qdrant vector store is included in `Agentic_RAG/qdrant_data/`. It comes from the [course repository](https://github.com/hamzafarooq/multi-agent-course).
4. Select the `.venv` kernel and use **Run All**. The first run downloads the embedding model (about 500 MB).

**Notes**

- When a SerpApi request fails, the error message includes the full request URL, and that URL contains `api_key=...`. Clear that output before sharing a notebook.
- If the last line of a cell returns a value, Jupyter displays it again. Use `_ = func()` to suppress the duplicate.
