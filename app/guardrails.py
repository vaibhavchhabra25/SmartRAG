"""Input/output safety guardrails: LLM-as-judge classifiers around the RAG pipeline."""
from dataclasses import dataclass

from app import config
from app.llm import call_json

REFUSAL_MESSAGES = {
    "jailbreak": "I can't follow instructions that try to override my operating guidelines. I'm here to answer questions about internal bank policy and procedure — happy to help with that.",
    "pii_request": "I don't have access to individual customer account data, and I can't look up or repeat specific customer information. Please use the core banking or case management system for that, where access is properly logged. I can help with general policy questions instead.",
    "advice_seeking": "I can't provide personalized investment, legal, or case-specific compliance advice. For a specific situation, please consult a Compliance officer or legal counsel. I can explain what the general policy says on this topic if that helps.",
    "off_domain": "I'm built to answer questions about internal bank compliance policy and procedure (AML, KYC, sanctions, transaction monitoring, data privacy). That question is outside my scope.",
    "harmful": "I can't help with that.",
}


@dataclass
class GuardrailResult:
    verdict: str  # "safe"/"unsafe" for input, "pass"/"fail" for output
    category: str
    reason: str

    @property
    def blocked(self) -> bool:
        return self.verdict in ("unsafe", "fail")


def _load_prompt(name: str) -> str:
    return (config.PROMPTS_DIR / name).read_text(encoding="utf-8")


def check_input(message: str) -> GuardrailResult:
    prompt = _load_prompt("input_guardrail.md").format(message=message)
    result = call_json(config.GUARDRAIL_MODEL, prompt)
    category = result.get("category", "off_domain")
    # Derive verdict from category ourselves rather than trusting the model's own (redundant)
    # verdict field — weaker/faster classifier models sometimes return inconsistent pairs
    # (e.g. category="safe" but verdict="unsafe").
    verdict = "safe" if category == "safe" else "unsafe"
    return GuardrailResult(verdict=verdict, category=category, reason=result.get("reason", ""))


def check_output(answer: str) -> GuardrailResult:
    prompt = _load_prompt("output_guardrail.md").format(answer=answer)
    result = call_json(config.GUARDRAIL_MODEL, prompt)
    category = result.get("category", "unsafe_content")
    verdict = "pass" if category == "none" else "fail"
    return GuardrailResult(verdict=verdict, category=category, reason=result.get("reason", ""))


def refusal_for(category: str) -> str:
    return REFUSAL_MESSAGES.get(category, "I'm not able to help with that request.")
