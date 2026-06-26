"""Central configuration for models, paths, and retrieval/guardrail tuning."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "policies"
CHROMA_DIR = ROOT_DIR / ".chroma"
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# "anthropic" (default, recommended) or "groq" (free-tier alternative, e.g. for local dev
# without Anthropic credits). Swap by setting LLM_PROVIDER and the matching *_API_KEY.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic" if ANTHROPIC_API_KEY else "groq")

_DEFAULT_MODELS = {
    "anthropic": {
        "guardrail": "claude-haiku-4-5-20251001",
        "judge": "claude-haiku-4-5-20251001",
        "answer": "claude-sonnet-4-6",
    },
    "groq": {
        "guardrail": "llama-3.1-8b-instant",
        "judge": "llama-3.1-8b-instant",
        "answer": "llama-3.3-70b-versatile",
    },
}

GUARDRAIL_MODEL = os.environ.get("GUARDRAIL_MODEL", _DEFAULT_MODELS[LLM_PROVIDER]["guardrail"])
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", _DEFAULT_MODELS[LLM_PROVIDER]["judge"])
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", _DEFAULT_MODELS[LLM_PROVIDER]["answer"])

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4

# Hierarchical retrieval: fetch a wider pool of child chunks, rerank, then expand the winners
# to their parent section text for generation.
RETRIEVAL_CANDIDATES = 12
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-base")
# Final score = (1 - RERANK_WEIGHT) * normalized_embedding_score + RERANK_WEIGHT * normalized_rerank_score.
# Blended rather than rerank-only: on this small, jargon-dense, structurally uniform corpus,
# general-purpose cross-encoders (tested both ms-marco-MiniLM-L-6-v2 and bge-reranker-base) can
# bury a clearly-correct section that embedding similarity already ranked highly. Blending lets
# the reranker break ties / catch lexical matches embeddings miss, without letting it fully
# override a strong embedding match.
RERANK_WEIGHT = float(os.environ.get("RERANK_WEIGHT", "0.4"))

# Cap on parent (section) size in characters, so even an unstructured long document can't blow
# up the generation context with one giant "section".
PARENT_MAX_CHARS = 3000
PARENT_STORE_PATH = CHROMA_DIR / "parents.json"

HALLUCINATION_GROUNDED_THRESHOLD = 0.85
COLLECTION_NAME = "compliance_policies"
