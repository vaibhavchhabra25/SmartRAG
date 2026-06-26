"""LangGraph pipeline wiring guardrails, retrieval, generation, and hallucination checks together."""
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app import config
from app.guardrails import check_input, check_output, refusal_for
from app.hallucination import STRICTER_INSTRUCTION, UNVERIFIED_WARNING, check_grounding
from app.llm import call_text
from app.vectorstore import similarity_search

CITATION_RE = re.compile(r"\[Source:\s*([^\]]+?)\s*§\s*([^\]]+?)\]")


class PipelineState(TypedDict, total=False):
    question: str
    blocked: bool
    block_reason: str
    retrieved: list
    context: str
    answer: str
    grounded_score: float
    unsupported_claims: list[str]
    regenerated: bool
    final_answer: str
    citations: list[dict]
    safety: dict


def input_guardrail_node(state: PipelineState) -> PipelineState:
    result = check_input(state["question"])
    if result.blocked:
        return {
            "blocked": True,
            "final_answer": refusal_for(result.category),
            "safety": {"stage": "input_guardrail", "category": result.category, "reason": result.reason},
        }
    return {"blocked": False}


def retrieve_node(state: PipelineState) -> PipelineState:
    hits = similarity_search(state["question"], k=config.TOP_K)
    retrieved = [{"text": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in hits]
    context = "\n\n".join(
        f"[source_doc: {r['metadata']['source_doc']} | section: {r['metadata']['section']}]\n{r['text']}"
        for r in retrieved
    )
    return {"retrieved": retrieved, "context": context}


def generate_node(state: PipelineState) -> PipelineState:
    prompt_template = (config.PROMPTS_DIR / "system_answer.md").read_text(encoding="utf-8")
    prompt = prompt_template.format(context=state["context"], question=state["question"])
    if state.get("regenerated"):
        prompt += STRICTER_INSTRUCTION
    answer = call_text(config.ANSWER_MODEL, prompt)
    return {"answer": answer}


def hallucination_node(state: PipelineState) -> PipelineState:
    grounding = check_grounding(state["answer"], state["context"])
    return {"grounded_score": grounding.grounded_score, "unsupported_claims": grounding.unsupported_claims}


def regenerate_flag_node(state: PipelineState) -> PipelineState:
    return {"regenerated": True}


def output_guardrail_node(state: PipelineState) -> PipelineState:
    answer = state["answer"]
    if state.get("grounded_score", 1.0) < config.HALLUCINATION_GROUNDED_THRESHOLD:
        answer = answer + UNVERIFIED_WARNING

    result = check_output(answer)
    if result.blocked:
        return {
            "blocked": True,
            "final_answer": refusal_for("harmful") if result.category == "unsafe_content" else (
                "I drafted a response but it didn't pass our safety review (it may have referenced "
                "specific customer data or given advice outside my scope), so I'm not able to show it. "
                "Please rephrase your question as a general policy question."
            ),
            "safety": {"stage": "output_guardrail", "category": result.category, "reason": result.reason},
        }

    citations = [
        {"source_doc": m.group(1).strip(), "section": m.group(2).strip()} for m in CITATION_RE.finditer(answer)
    ]
    return {
        "final_answer": answer,
        "citations": citations,
        "safety": {
            "stage": "passed",
            "grounded_score": state.get("grounded_score", 1.0),
            "regenerated": state.get("regenerated", False),
        },
    }


def route_after_input(state: PipelineState) -> str:
    return "blocked" if state.get("blocked") else "continue"


def route_after_hallucination(state: PipelineState) -> str:
    if state.get("grounded_score", 1.0) < config.HALLUCINATION_GROUNDED_THRESHOLD and not state.get("regenerated"):
        return "regenerate"
    return "continue"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("input_guardrail", input_guardrail_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("hallucination_check", hallucination_node)
    graph.add_node("flag_regenerated", regenerate_flag_node)
    graph.add_node("output_guardrail", output_guardrail_node)

    graph.set_entry_point("input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail", route_after_input, {"blocked": END, "continue": "retrieve"}
    )
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "hallucination_check")
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {"regenerate": "flag_regenerated", "continue": "output_guardrail"},
    )
    graph.add_edge("flag_regenerated", "generate")
    graph.add_edge("output_guardrail", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def ask(question: str) -> PipelineState:
    return get_graph().invoke({"question": question})
