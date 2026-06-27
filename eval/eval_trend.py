"""Render eval/{ragas,guardrail}_history.jsonl into a trend report, flagging regressions.

Pure read-over-existing-history, zero LLM calls — safe to run anytime. Run with:
python -m eval.eval_trend
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.history import load_history  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "eval_trend.md"

# Drop below previous run by more than this and it's flagged as a regression, not just noise.
REGRESSION_TOLERANCE = 0.05

NUMERIC_KEYS = {
    "ragas": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
    "guardrail": ["block_rate", "fp_rate"],
}
# Higher is better for everything except fp_rate (lower is better).
LOWER_IS_BETTER = {"fp_rate"}


def _trend_table(name: str, records: list[dict]) -> list[str]:
    keys = [k for k in NUMERIC_KEYS[name] if records and k in records[0]]
    if not records:
        return [f"## {name}", "", "No history yet — run the eval at least once.", ""]

    lines = [f"## {name}", "", f"| Run | Git SHA | {' | '.join(keys)} |", "|---|---|" + "---|" * len(keys)]
    prev = None
    regressions = []
    for i, rec in enumerate(records):
        cells = []
        for k in keys:
            value = rec.get(k)
            cell = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
            if prev is not None and k in prev and isinstance(value, (int, float)):
                delta = value - prev[k]
                regressed = (delta < -REGRESSION_TOLERANCE) if k not in LOWER_IS_BETTER else (
                    delta > REGRESSION_TOLERANCE
                )
                if regressed:
                    cell += " ⚠️"
                    regressions.append((i, k, prev[k], value))
                sign = "+" if delta >= 0 else ""
                cell += f" ({sign}{delta:.3f})"
            cells.append(cell)
        lines.append(f"| {i + 1} ({rec['timestamp'][:19]}) | {rec.get('git_sha', '?')} | {' | '.join(cells)} |")
        prev = rec

    if regressions:
        lines += ["", f"**{len(regressions)} regression(s) flagged** (drop > {REGRESSION_TOLERANCE} vs previous run):"]
        for i, k, old, new in regressions:
            lines.append(f"- Run {i + 1}: `{k}` {old:.3f} -> {new:.3f}")
    else:
        lines += ["", "No regressions flagged."]
    lines.append("")
    return lines


def build_report() -> str:
    lines = ["# Eval Trend", ""]
    for name in ("ragas", "guardrail"):
        lines.extend(_trend_table(name, load_history(name)))
    return "\n".join(lines)


if __name__ == "__main__":
    report = build_report()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nFull report written to {REPORT_PATH}")
