"""Lightweight run logging: one JSON line per question, no extra LLM calls.

This is the observability piece of the pipeline — every ask() call is recorded with enough
detail (latency, model config, safety outcome, grounding score, citation count) to debug a bad
answer after the fact or compute aggregate stats over time, without needing an external tracing
service. See eval/runs_summary.py for turning these logs into aggregate numbers.
"""
import json
from datetime import datetime, timezone

from app import config

LOG_PATH = config.ROOT_DIR / "logs" / "runs.jsonl"


def log_run(question: str, result: dict, latency_s: float) -> None:
    safety = result.get("safety", {})
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "latency_s": round(latency_s, 3),
        "llm_provider": config.LLM_PROVIDER,
        "answer_model": config.ANSWER_MODEL,
        "guardrail_model": config.GUARDRAIL_MODEL,
        "blocked": bool(result.get("blocked")),
        "safety_stage": safety.get("stage"),
        "safety_category": safety.get("category"),
        "grounded_score": result.get("grounded_score"),
        "regenerated": bool(result.get("regenerated")),
        "num_citations": len(result.get("citations") or []),
        "num_retrieved": len(result.get("retrieved") or []),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_runs() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with open(LOG_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
