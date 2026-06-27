"""Append-only eval score history, so quality is a trend over commits, not a single snapshot.

eval/run_ragas_eval.py and eval/guardrail_eval.py each append one record per run to their own
history file (e.g. eval/ragas_history.jsonl). Unlike eval/report.md (overwritten every run,
gitignored), these history files are meant to be committed — they're the actual regression
record: "did faithfulness drop after that prompt change?" Reading them costs nothing extra (no
LLM calls); appending costs nothing extra either, it's just persisting a result the eval already
computed.
"""
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HISTORY_DIR = Path(__file__).resolve().parent


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=HISTORY_DIR,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def append_history(name: str, metrics: dict) -> None:
    """Append one record (timestamp + git sha + given metrics) to eval/{name}_history.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        **metrics,
    }
    path = HISTORY_DIR / f"{name}_history.jsonl"
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_history(name: str) -> list[dict]:
    path = HISTORY_DIR / f"{name}_history.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
