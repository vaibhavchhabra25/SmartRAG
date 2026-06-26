# Architecture & Design Notes

This document explains *what Compliance Copilot is, what problem it solves, and which RAG
concepts and frameworks it's built from* — the technical depth behind the quickstart in the
[README](README.md).

## What it is

Compliance Copilot is an internal knowledge assistant for a bank's Risk & Compliance team. An
employee asks a policy question in plain English; the system answers strictly from the bank's
own internal documents (AML, KYC, sanctions, transaction monitoring, data privacy) and attaches
a citation to the exact policy section each claim came from.

## What it solves

A generic LLM chatbot is a liability in a regulated domain for three concrete reasons, and the
system is built specifically to close each one:

1. **Hallucinated rules.** An LLM asked "what's our cash reporting threshold?" will confidently
   answer from its training data about banking regulation in general — which may not match
   *this* bank's actual internal policy, and a wrong number cited as policy is a compliance
   incident. → Solved by **retrieval-grounded generation + a hallucination judge** (below).
2. **Unsafe requests the assistant shouldn't fulfil.** Employees will ask it things it has no
   business answering: a specific customer's account balance, "should we file a SAR on this
   customer," investment advice, or attempts to jailbreak it into ignoring its scope. → Solved
   by **input/output guardrails**.
3. **No way to know if it's actually working.** Without measurement, "the bot seems fine" is not
   a basis for shipping something compliance-sensitive. → Solved by a **RAGAS-based quality eval
   + a separate guardrail-effectiveness eval**, both runnable as CI gates.

## RAG pipeline, end to end

```
question
  │
  ▼
[input guardrail]  classifies: safe / jailbreak / PII-request / advice-seeking / off-domain / harmful
  │
  ├─ unsafe ──────────────────────────────────────────► refusal message (with reason) ─► done
  ▼ safe
[retrieve]            wide candidate search → blended rerank → expand winners to parent sections
  ▼
[generate]            LLM answers using ONLY retrieved context, with inline [Source: doc §section] citations
  ▼
[hallucination check] judge scores how well the answer's claims are grounded in retrieved context
  │
  ├─ ungrounded (score < 0.85) and not yet retried ──► regenerate once, stricter "stick to context" prompt
  ▼ grounded (or already retried)
[output guardrail]    checks final answer for leaked PII / advice-giving / unsafe content
  │
  ├─ fails ───────────────────────────────────────────► safe fallback message
  ▼ passes
respond (answer + structured citations + safety metadata)
```

Each box is a node in a `langgraph.StateGraph` ([app/graph.py](app/graph.py)); the arrows with
conditions are conditional edges. The state object threaded through every node
(`PipelineState`) is a plain `TypedDict` — no hidden agent memory, no implicit context.

### Why a graph instead of a linear chain

The pipeline isn't a straight line — it has two branch points (block on input, retry on
ungrounded output) and one loop (regenerate-once). LangGraph models that directly as nodes +
conditional edges (`add_conditional_edges`) instead of nesting if/else inside a single chain
function, which is what makes the retry-once and early-exit-on-block logic legible in
[app/graph.py](app/graph.py) rather than buried in control flow.

## RAG concepts, mapped to code

| Concept | What it does here | Where |
|---|---|---|
| **Hierarchical chunking (parent sections / child chunks)** | A 3-tier fallback resolves each document's "sections": real `## §N Title` headers (the policy convention) → generic markdown headers (for structured uploads) → fixed-size pseudo-sections labeled "Part N" (for fully unstructured long text, e.g. a plain PDF with no headings). Every resolved section is capped at `PARENT_MAX_CHARS` so even a real `§N` section can't blow up the context window for a long document. Each capped section becomes a **parent**; it's then split into small (~800 char) **child** chunks for embedding, each tagged with its `parent_id`. Short docs end up ~1 child per section (same behavior as before); long docs get many children per parent, all traceable back to one bounded, citable section. | [app/ingest.py](app/ingest.py) `_split_into_sections`, `chunk_documents` |
| **Parent store** | Children are the embedding/search unit; parents are looked up by `parent_id` at query time to give generation full section context instead of an isolated fragment. Persisted as a small JSON file (`.chroma/parents.json`) rather than kept in memory, since the CLI is a fresh process per question — ingest and query happen in separate process runs. | [app/parent_store.py](app/parent_store.py) |
| **Embeddings** | Local `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` — no embedding API key, runs offline, fast enough for a knowledge base this size. Swappable via `EMBEDDING_MODEL` in `.env`. | [app/vectorstore.py](app/vectorstore.py) |
| **Vector store** | Chroma, persisted to a local directory (`.chroma/`). One collection, similarity search with relevance scores over child chunks. | [app/vectorstore.py](app/vectorstore.py) |
| **Reranking (blended)** | Retrieval first casts a wide net (`RETRIEVAL_CANDIDATES`, 12 child chunks by embedding similarity), then a cross-encoder (`BAAI/bge-reranker-base`) scores each `(question, chunk)` pair directly. The cross-encoder score is **blended** with the original embedding score (`RERANK_WEIGHT`, default 0.4 toward the cross-encoder) rather than used alone — see the limitations section for why pure reranking was tried and rejected. | [app/rerank.py](app/rerank.py) |
| **Hierarchical expansion** | The top `TOP_K` (4) reranked child chunks are deduped by `parent_id` (a long document can surface several children from the same section) and expanded to their full parent section text via the parent store — this is what gives long documents enough surrounding context instead of a disconnected 800-char fragment. | `retrieve_node` in [app/graph.py](app/graph.py) |
| **Grounded generation** | The answer prompt explicitly forbids using anything but the retrieved context and requires inline `[Source: doc §section]` tags — citation isn't a post-hoc add-on, it's part of what the model is instructed to produce. | [app/prompts/system_answer.md](app/prompts/system_answer.md) |
| **Citation extraction** | A regex (`CITATION_RE`) parses the model's own `[Source: ...]` tags out of the answer text into a structured `citations` list, so the UI/CLI render them separately from prose. Unaffected by the parent expansion — citations are still `(source_doc, section)`, and parent = section. | `output_guardrail_node` in [app/graph.py](app/graph.py) |
| **Incremental indexing** | New documents (uploaded via the UI or dropped into `data/policies/`) can be embedded and added to the *existing* collection (children + parents) without re-embedding everything else — `ingest_file()` vs the full `build_index()` rebuild. | [app/ingest.py](app/ingest.py) |

## Safety layer

Two LLM-as-judge classifiers sit around the retrieval/generation core, each a single
purpose-built prompt that returns strict JSON (`{"category": ..., "reason": ...}`) — the code
derives a boolean `blocked` from `category` itself rather than trusting a separate verdict field
from the model (a real bug surfaced during testing with a smaller model: the two fields could
disagree).

- **Input guardrail** ([app/prompts/input_guardrail.md](app/prompts/input_guardrail.md)):
  classifies the *question* before any retrieval happens — `safe`, `jailbreak`, `pii_request`,
  `advice_seeking`, `off_domain`, or `harmful`. Blocking here means zero retrieval or generation
  cost is spent on a request that was never going to be answered.
- **Output guardrail** ([app/prompts/output_guardrail.md](app/prompts/output_guardrail.md)):
  classifies the *drafted answer* — catches a model accidentally inventing/repeating customer
  PII, giving case-specific advice, or producing unsafe content, even if the input looked benign.
- Each category maps to a specific, honest refusal message ([app/guardrails.py](app/guardrails.py))
  rather than a generic "I can't help with that" — e.g. the PII refusal explains *why* and points
  to the right system to use instead.

## Hallucination mitigation

This is the part that matters most in a regulated domain: an answer that *sounds* right but
states a policy detail the documents don't actually support.

[app/hallucination.py](app/hallucination.py) implements a **separate judge call**
([app/prompts/hallucination_judge.md](app/prompts/hallucination_judge.md)) that decomposes the
drafted answer into individual factual claims and checks each one against the retrieved context
directly — not against general knowledge, not against plausibility. It returns a
`grounded_score` (supported claims / total claims) and a list of unsupported claims.

- **Score ≥ 0.85**: answer ships as-is.
- **Score < 0.85, first attempt**: the graph loops back to `generate` with an added instruction
  to restate only what the context says — a single retry, not an open-ended loop.
- **Still ungrounded after retry**: the answer ships with a visible "⚠️ unverified" warning
  rather than being silently presented as settled fact. Compliance is a domain where "I'm not
  sure" is a better failure mode than a confident wrong answer.

## Quality evaluation

Two independent eval scripts, because RAGAS and "did the guardrail do its job" measure different
things:

- **[eval/run_ragas_eval.py](eval/run_ragas_eval.py)** runs the *entire* pipeline (not a stub)
  over 15 hand-written policy questions with reference answers
  ([eval/golden_dataset.json](eval/golden_dataset.json)), then scores the results with
  [RAGAS](https://docs.ragas.io/):
  - `faithfulness` — are the answer's claims supported by the retrieved context (a second,
    independent measurement of the same property the in-pipeline hallucination judge checks —
    useful precisely because it's computed by a different prompt/process, catching cases the
    inline judge might miss).
  - `answer_relevancy` — does the answer actually address the question asked.
  - `context_precision` / `context_recall` — did retrieval surface the right chunks, and did it
    surface *enough* of them.
- **[eval/guardrail_eval.py](eval/guardrail_eval.py)** measures what RAGAS can't: refusal
  correctness. It runs 10 adversarial prompts (jailbreak, PII, advice-seeking, off-domain,
  harmful) and checks the guardrail blocks them with the right category, plus runs the 15
  legitimate questions to measure the false-positive rate (don't want a guardrail so aggressive
  it blocks real policy questions).
- **[tests/test_eval.py](tests/test_eval.py)** wraps both with pytest assertions against fixed
  thresholds (`faithfulness ≥ 0.80`, adversarial block rate ≥ 80%, false-positive rate ≤ 10%),
  so a regression fails a test run instead of just looking worse in a markdown report.

## Frameworks and why

| Tool | Role | Why this one |
|---|---|---|
| **LangGraph** | Orchestrates the pipeline as a state graph with conditional branching | The pipeline has real branch points (block/continue, retry/pass) — a graph makes that explicit instead of nested conditionals in a chain |
| **LangChain** | LLM/embeddings/vector-store interfaces, text splitting | Common interface (`ChatAnthropic`/`ChatGroq`, `HuggingFaceEmbeddings`, `Chroma`) so swapping providers is a config change, not a rewrite |
| **Claude (Anthropic) / Groq** | The LLM doing classification, generation, and judging | Claude is the default/recommended path (`app/config.py`); Groq's free tier is a pluggable fallback for local dev without Anthropic credits — same code, `LLM_PROVIDER` env var picks the client |
| **Chroma** | Vector store | Local, zero-infra, persists to disk — appropriate for a project-scoped knowledge base; would swap for a managed store at real scale |
| **sentence-transformers (`all-MiniLM-L6-v2`)** | Embeddings | Runs locally, no extra API key, fast enough for this corpus size |
| **sentence-transformers `CrossEncoder` (`BAAI/bge-reranker-base`)** | Reranking | Local, no API key; used as one signal blended with embedding similarity rather than the sole ranker (see Known Limitations — pure reranking underperformed here) |
| **RAGAS** | RAG quality metrics (faithfulness, relevancy, precision, recall) | Purpose-built for exactly this — avoids hand-rolling LLM-judge eval prompts for metrics that already have a maintained, citable implementation |
| **Streamlit** | Chat UI + document upload | Fastest path to an interactive demo with file upload, chat history, and expandable citation/safety panels, with minimal code |
| **pypdf** | PDF text extraction for uploaded documents | Lets the upload feature accept real policy PDFs, not just markdown |
| **pytest** | Test runner | Standard; `pytest.ini` marks the eval-backed tests `slow` since they make real LLM calls |

## Known limitations / deliberate scope cuts

- **Pure cross-encoder reranking was tried and didn't work well here.** Two general-purpose
  rerankers (`cross-encoder/ms-marco-MiniLM-L-6-v2`, then `BAAI/bge-reranker-base`) were tested
  as the sole ranking signal. Both sometimes buried a section that embedding similarity had
  already ranked near the top — e.g. for "What are the KYC requirements for opening a new
  corporate account?", the correct "§2 Corporate Customer Identification" section was embedding
  rank #2 (0.39, essentially tied for #1) but cross-encoder rank #5-6 out of 12 candidates. On a
  small, jargon-dense, structurally uniform policy corpus, these web-search-trained rerankers
  don't transfer cleanly. The fix was to **blend** the cross-encoder score with the original
  embedding score (`RERANK_WEIGHT`, [app/rerank.py](app/rerank.py)) rather than trust either
  alone — worth knowing if you swap in a different reranker or corpus: re-validate before
  assuming a "better" reranker model fixes ranking quality on its own.
- **No query rewriting/decomposition** — a single embedding pass per question; a multi-part
  question ("what's the AML threshold and how does it differ from KYC review frequency?") isn't
  split into sub-queries.
- **No multi-turn memory in the pipeline itself** — the Streamlit UI keeps chat history for
  display, but each question is answered independently; there's no conversational
  follow-up resolution ("what about for PEPs?" referring to the prior turn).
- **Synthetic policy documents** — realistic in structure and content but written for this
  project, not pulled from a real bank's actual policies.
- **Single regeneration attempt** on a failed grounding check, not an open-ended refinement
  loop — a deliberate bound on latency/cost versus chasing a perfect score.
- **Uploaded documents aren't deduplicated** — re-uploading the same file twice adds duplicate
  chunks; acceptable for a demo, would need a content hash check for production use.
- **Pseudo-section fallback is positional, not semantic** — an unstructured long upload gets cut
  into fixed-size "Part N" sections by character count, not by topic boundary. It bounds context
  size safely, but a real outline (or a semantic chunker) would produce more coherent sections
  for a genuinely long, unstructured document.
