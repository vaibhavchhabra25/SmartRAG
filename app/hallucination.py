"""Grounding/hallucination check: verifies the answer's claims against retrieved context."""
from dataclasses import dataclass

from app import config
from app.llm import call_json

STRICTER_INSTRUCTION = (
    "\n\nIMPORTANT: Your previous answer included claims not supported by the context. "
    "Re-answer using ONLY sentences that restate or directly summarize the retrieved context below. "
    "If something is not explicitly stated in the context, omit it or say you don't have enough information."
)


@dataclass
class GroundingResult:
    grounded_score: float
    unsupported_claims: list[str]

    @property
    def is_grounded(self) -> bool:
        return self.grounded_score >= config.HALLUCINATION_GROUNDED_THRESHOLD


def _load_prompt() -> str:
    return (config.PROMPTS_DIR / "hallucination_judge.md").read_text(encoding="utf-8")


def check_grounding(answer: str, context: str) -> GroundingResult:
    prompt = _load_prompt().format(context=context, answer=answer)
    result = call_json(config.JUDGE_MODEL, prompt)
    return GroundingResult(
        grounded_score=float(result.get("grounded_score", 0.0)),
        unsupported_claims=result.get("unsupported_claims", []),
    )


UNVERIFIED_WARNING = (
    "\n\n⚠️ Note: parts of this answer could not be fully verified against the policy "
    "knowledge base. Please confirm with the source documents or Compliance before relying on it."
)
