"""Summarize logged pipeline runs (logs/runs.jsonl) into aggregate operational stats.

This is read-only over existing logs — it makes zero LLM calls, so it's safe to run as often as
you like even on a rate-limited API key. Run with: python -m eval.runs_summary
"""
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.telemetry import read_runs  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent / "runs_summary.md"


def summarize(runs: list[dict]) -> str:
    if not runs:
        return "# Runs Summary\n\nNo logged runs yet — ask some questions first (CLI or UI), then rerun this."

    total = len(runs)
    blocked = [r for r in runs if r["blocked"]]
    passed = [r for r in runs if not r["blocked"]]
    regenerated = [r for r in runs if r.get("regenerated")]
    grounded_scores = [r["grounded_score"] for r in runs if r.get("grounded_score") is not None]
    latencies = [r["latency_s"] for r in runs if r.get("latency_s") is not None]

    category_counts: dict[str, int] = {}
    for r in blocked:
        cat = r.get("safety_category") or "unknown"
        category_counts[cat] = category_counts.get(cat, 0) + 1

    lines = [
        "# Runs Summary",
        "",
        f"Total logged runs: **{total}**",
        f"- Blocked by guardrails: {len(blocked)} ({len(blocked) / total:.0%})",
        f"- Answered: {len(passed)} ({len(passed) / total:.0%})",
        f"- Regenerated due to low grounding: {len(regenerated)} ({len(regenerated) / total:.0%})",
    ]
    if latencies:
        lines.append(f"- Avg latency: {mean(latencies):.2f}s (min {min(latencies):.2f}s, max {max(latencies):.2f}s)")
    if grounded_scores:
        lines.append(f"- Avg grounded score (answered runs): {mean(grounded_scores):.3f}")

    if category_counts:
        lines += ["", "## Blocked-run breakdown by category", "", "| Category | Count |", "|---|---|"]
        for cat, count in sorted(category_counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {cat} | {count} |")

    lines += ["", "## Most recent 10 runs", "", "| Time | Latency | Blocked | Category | Grounded |", "|---|---|---|---|---|"]
    for r in runs[-10:]:
        lines.append(
            f"| {r['timestamp']} | {r.get('latency_s', '-')}s | {r['blocked']} | "
            f"{r.get('safety_category', '-')} | {r.get('grounded_score', '-')} |"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    report = summarize(read_runs())
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nFull report written to {REPORT_PATH}")
