"""Hallucination/grounding judge tests. Requires ANTHROPIC_API_KEY."""
from app.hallucination import check_grounding

CONTEXT = (
    "[source_doc: Anti-Money Laundering (AML) Policy | section: §5 Cash Transaction Thresholds]\n"
    "§5 Cash Transaction Thresholds\nAny single cash transaction of $10,000 or more, or multiple related cash "
    "transactions that aggregate to $10,000 or more within a single business day, must be reported via a "
    "Currency Transaction Report (CTR), regardless of whether the activity appears suspicious."
)


def test_grounded_answer_scores_high():
    answer = (
        "Cash transactions of $10,000 or more (single or aggregated within a business day) must be reported "
        "via a Currency Transaction Report [Source: Anti-Money Laundering (AML) Policy §5 Cash Transaction "
        "Thresholds]."
    )
    result = check_grounding(answer, CONTEXT)
    assert result.is_grounded
    assert result.grounded_score >= 0.85


def test_ungrounded_answer_scores_low():
    answer = (
        "Cash transactions of $25,000 or more must be reported, and the report must be filed within 48 hours "
        "to the Federal Reserve [Source: Anti-Money Laundering (AML) Policy §5 Cash Transaction Thresholds]."
    )
    result = check_grounding(answer, CONTEXT)
    assert not result.is_grounded
    assert len(result.unsupported_claims) > 0
