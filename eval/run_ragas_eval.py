"""Run the full RAG pipeline over the golden dataset and score it with RAGAS.

Run with: python -m eval.run_ragas_eval
Writes eval/report.md with per-question and aggregate scores.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ragas==0.4.x unconditionally imports langchain_community.chat_models.vertexai at module load
# time (ragas/llms/base.py), even though we never use Vertex AI. That submodule was removed from
# recent langchain-community releases (which we need for compatibility with langchain-core 1.x),
# so the import fails. Stub it out rather than pinning to an older, conflicting langchain-community.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ModuleNotFoundError:
        import types

        stub = types.ModuleType("langchain_community.chat_models.vertexai")
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = stub

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from app import config
from app.graph import ask
from app.llm import get_client
from app.vectorstore import get_embeddings
from eval.history import append_history

DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"
REPORT_PATH = Path(__file__).resolve().parent / "report.md"

THRESHOLDS = {
    "faithfulness": 0.80,
    "answer_relevancy": 0.70,
    "context_precision": 0.60,
    "context_recall": 0.60,
}


def collect_samples() -> list[SingleTurnSample]:
    data = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    samples = []
    for item in data["normal"]:
        result = ask(item["question"])
        contexts = [r["text"] for r in result.get("retrieved", [])]
        samples.append(
            SingleTurnSample(
                user_input=item["question"],
                response=result.get("final_answer", ""),
                retrieved_contexts=contexts or [""],
                reference=item["ground_truth"],
            )
        )
    return samples


def run():
    samples = collect_samples()
    dataset = EvaluationDataset(samples=samples)

    judge_llm = get_client(config.JUDGE_MODEL, temperature=0)
    embeddings = get_embeddings()

    # strictness=1 (default 3) avoids AnswerRelevancy requesting n>1 completions per call, which
    # Groq's API rejects ("'n' must be at most 1"); harmless on providers that do support it.
    metrics = [
        Faithfulness(),
        AnswerRelevancy(strictness=1),
        ContextPrecision(),
        ContextRecall(),
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()
    means = {col: df[col].mean() for col in THRESHOLDS if col in df.columns}

    lines = ["# RAGAS Evaluation Report", "", "## Aggregate scores", ""]
    lines.append("| Metric | Score | Threshold | Status |")
    lines.append("|---|---|---|---|")
    all_pass = True
    for metric, threshold in THRESHOLDS.items():
        score = means.get(metric)
        if score is None:
            continue
        status = "PASS" if score >= threshold else "FAIL"
        all_pass = all_pass and score >= threshold
        lines.append(f"| {metric} | {score:.3f} | {threshold:.2f} | {status} |")

    lines += ["", "## Per-question detail", ""]
    for _, row in df.iterrows():
        lines.append(f"### {row['user_input']}")
        lines.append(f"- response: {row['response'][:300]}")
        for metric in THRESHOLDS:
            if metric in row:
                lines.append(f"- {metric}: {row[metric]:.3f}")
        lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:8]))
    print(f"\nFull report written to {REPORT_PATH}")

    append_history("ragas", {**{k: round(float(v), 4) for k, v in means.items()}, "all_pass": all_pass})

    return means, all_pass


if __name__ == "__main__":
    _, all_pass = run()
    sys.exit(0 if all_pass else 1)
