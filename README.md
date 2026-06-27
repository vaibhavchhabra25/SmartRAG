# Compliance Copilot

An internal knowledge assistant for a bank's Risk & Compliance team. Ask it about AML, KYC,
sanctions screening, transaction monitoring, or data privacy policy and it answers **only**
from the bank's own policy documents, with inline citations — and it actively refuses to give
investment/legal advice, leak customer PII, or get jailbroken into ignoring its scope.

Built with **LangGraph + Claude + Chroma**, with a dedicated safety layer (input/output
guardrails + a hallucination/grounding judge) and a **RAGAS**-based quality eval suite.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the deeper write-up: what problem each piece solves,
how the RAG concepts map to code, and why each framework was chosen.

## Why this domain

Risk & compliance is a great stress test for a "knowledge + citations" assistant: the cost of a
hallucinated policy rule or a leaked customer detail is high, the org has a lot of internal
procedure to encode, and there's a natural set of things the assistant must refuse to do
(case-specific advice, PII lookups, prompt injection). The sample knowledge base
(`data/policies/`) contains six synthetic-but-realistic policies: AML, KYC, sanctions
screening, transaction monitoring, data privacy, and a regulatory FAQ.

## Architecture

```
question
  │
  ▼
[input guardrail]  Claude classifies: safe / jailbreak / PII-request / advice-seeking / off-domain / harmful
  │
  ├─ unsafe ──────────────────────────────────────────► refusal message (with reason) ─► done
  │
  ▼ safe
[retrieve]            wide candidate search (Chroma) → blended cross-encoder rerank → expand to parent sections
  │
  ▼
[generate]            Claude answers using ONLY retrieved context, with inline [Source: doc §section] citations
  │
  ▼
[hallucination check] Claude judge scores how well the answer's claims are grounded in the retrieved context
  │
  ├─ ungrounded (score < 0.85) and not yet retried ──► regenerate once with a stricter "stick to context" prompt
  │
  ▼ grounded (or already retried)
[output guardrail]    Claude checks the final answer for leaked PII / advice-giving / unsafe content
  │
  ├─ fails ───────────────────────────────────────────► safe fallback message
  │
  ▼ passes
respond (answer + structured citations + safety metadata)
```

This is implemented as a `langgraph` `StateGraph` in [app/graph.py](app/graph.py) — see
`input_guardrail_node`, `retrieve_node`, `generate_node`, `hallucination_node`, and
`output_guardrail_node`, wired together with conditional edges.

- **Hierarchical retrieval**: documents are split into parent sections (`§N` headers → markdown
  headers → fixed-size pseudo-sections, whichever applies, each capped in size) and then into
  small child chunks for embedding. Retrieval fetches a wide pool of child chunks by embedding
  similarity, reranks them with a local cross-encoder *blended* with the embedding score, then
  expands the winners to their full parent section text for generation — good precision on short
  docs, good context completeness on long ones. See [ARCHITECTURE.md](ARCHITECTURE.md) for why
  pure cross-encoder reranking was tried and replaced with the blend. ([app/ingest.py](app/ingest.py),
  [app/rerank.py](app/rerank.py), [app/parent_store.py](app/parent_store.py))
- **Guardrails**: two LLM-as-judge classifiers ([app/guardrails.py](app/guardrails.py),
  prompts in [app/prompts/](app/prompts/)) — one on the incoming question, one on the drafted
  answer — each returning structured JSON (`verdict`, `category`, `reason`) so the graph can
  branch deterministically.
- **Hallucination/grounding check** ([app/hallucination.py](app/hallucination.py)): a separate
  Claude call breaks the answer into claims and checks each against the retrieved context,
  producing a `grounded_score`. Below threshold, the pipeline retries once with a stricter
  prompt; if still ungrounded, the answer ships with a visible "unverified" warning rather than
  being shown as settled fact.
- **Models**: fast/cheap `claude-haiku-4-5` for guardrail and judge calls, `claude-sonnet-4-6`
  for the actual answer generation ([app/config.py](app/config.py)).

## Quality evals

- [eval/golden_dataset.json](eval/golden_dataset.json): ~25 hand-written cases — 15 normal
  policy questions with reference answers, plus 10 adversarial prompts (jailbreak, PII
  requests, advice-seeking, off-domain, harmful) labeled with the guardrail category they
  should trigger.
- [eval/run_ragas_eval.py](eval/run_ragas_eval.py): runs the **full pipeline** over the normal
  questions and scores it with [RAGAS](https://docs.ragas.io/) — `faithfulness`,
  `answer_relevancy`, `context_precision`, `context_recall` — writing `eval/report.md`.
- [eval/guardrail_eval.py](eval/guardrail_eval.py): RAGAS doesn't cover refusal correctness, so
  this scripted eval measures the guardrail's block-rate on adversarial prompts and its
  false-positive rate on legitimate ones, writing `eval/guardrail_report.md`.
- [tests/test_eval.py](tests/test_eval.py) wraps both evals in pytest with hard thresholds
  (e.g. faithfulness ≥ 0.80) so a quality regression fails CI, not just looks bad in a report.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY (or a free Groq key — see .env.example)

python -m app.ingest    # build the Chroma index from data/policies/
```

## Try it

```bash
# CLI
python -m app.cli "What are the KYC requirements for opening a new corporate account?"
python -m app.cli "Should I buy Tesla stock?"          # refused: advice-seeking
python -m app.cli "What's John Smith's account balance?"  # refused: PII request

# Chat UI
streamlit run ui/streamlit_app.py
```

The chat UI's sidebar also lets you upload a new document (`.md`, `.txt`, or `.pdf`) straight
into the knowledge base — it's indexed immediately and searchable on the next question, without
rebuilding the rest of the index. See `ingest_file()` in [app/ingest.py](app/ingest.py).

## Run the evals

```bash
pytest tests/                       # unit tests for guardrails + hallucination judge
python -m eval.run_ragas_eval       # RAGAS quality report -> eval/report.md, appends to eval/ragas_history.jsonl
python -m eval.guardrail_eval       # guardrail report -> eval/guardrail_report.md, appends to eval/guardrail_history.jsonl
pytest tests/test_eval.py           # same two evals, asserted as CI gates
python -m eval.eval_trend           # trend across all runs so far -> eval/eval_trend.md, flags regressions
```

`eval/*_history.jsonl` is committed to the repo on purpose (unlike the other reports, which are
regenerated each run and gitignored) — it's the quality record across commits, not just a single
snapshot. Commit the updated history file after a real eval run if you want it on the record.

## Observability & CI

Every `ask()` call (CLI or UI) logs one line to `logs/runs.jsonl` — latency, guardrail
stage/category, grounded score, regeneration, citation count — with zero extra LLM calls (see
[app/telemetry.py](app/telemetry.py)). Turn it into aggregate stats anytime:

```bash
python -m eval.runs_summary         # block rate, avg latency/grounded score -> eval/runs_summary.md
```

CI is split in two, on purpose, because this project is tested against a free-tier rate-limited
key:
- **[.github/workflows/ci.yml](.github/workflows/ci.yml)** — every push/PR, zero LLM calls
  (pure-logic chunking tests + an index rebuild).
- **[.github/workflows/llm-eval.yml](.github/workflows/llm-eval.yml)** — manual trigger only
  (`workflow_dispatch`); runs the guardrail/hallucination tests and the full RAGAS/guardrail
  eval. Add `GROQ_API_KEY` or `ANTHROPIC_API_KEY` as a repo secret to use it.

See [ARCHITECTURE.md](ARCHITECTURE.md)'s LLMOps section for why it's split this way.

## Project layout

```
data/policies/       sample policy markdown docs (AML, KYC, sanctions, monitoring, privacy, FAQ)
app/
  config.py            models, paths, thresholds
  ingest.py             load -> section-split -> chunk -> embed -> Chroma + parent store
  vectorstore.py         Chroma wrapper
  parent_store.py         parent (section) text lookup, persisted to .chroma/parents.json
  rerank.py                cross-encoder rerank blended with embedding score
  llm.py                  Claude/Groq call helpers (plain text + strict JSON)
  guardrails.py            input/output safety classifiers
  hallucination.py          grounding judge
  telemetry.py              run logging (logs/runs.jsonl), zero extra LLM calls
  graph.py                   LangGraph pipeline
  cli.py                      quick CLI
  prompts/                     system + guardrail + judge prompts
ui/streamlit_app.py    chat UI + document upload
eval/                  golden dataset + RAGAS eval + guardrail eval + runs_summary + eval_trend
  history.py             append-only score history per eval (eval/*_history.jsonl, committed)
tests/                 pytest unit tests + eval threshold gates
.github/workflows/     ci.yml (free, every push) + llm-eval.yml (manual, real LLM calls)
```
