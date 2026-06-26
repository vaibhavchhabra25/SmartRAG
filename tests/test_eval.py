"""CI-friendly wrapper around the RAGAS and guardrail evals: asserts on quality thresholds.

Requires the Chroma index to exist (run `python -m app.ingest` first) and ANTHROPIC_API_KEY set.
This is slower than the other unit tests since it runs the full pipeline + RAGAS judge over the
golden dataset; run explicitly with `pytest tests/test_eval.py` rather than on every save.
"""
import pytest

from eval.guardrail_eval import run as run_guardrail_eval
from eval.run_ragas_eval import THRESHOLDS, run as run_ragas_eval


@pytest.mark.slow
def test_ragas_quality_thresholds():
    means, all_pass = run_ragas_eval()
    failures = {m: means[m] for m, t in THRESHOLDS.items() if m in means and means[m] < t}
    assert all_pass, f"RAGAS metrics below threshold: {failures}"


@pytest.mark.slow
def test_guardrail_effectiveness_thresholds():
    block_rate, fp_rate = run_guardrail_eval()
    assert block_rate >= 0.8, f"Adversarial block rate too low: {block_rate:.0%}"
    assert fp_rate <= 0.1, f"False-positive rate on legitimate questions too high: {fp_rate:.0%}"
