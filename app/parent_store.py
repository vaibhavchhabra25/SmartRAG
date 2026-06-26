"""Persisted lookup from parent_id -> full parent (section) text and metadata.

Child chunks are the retrieval/embedding unit (small, precise); each carries a parent_id back to
the section it came from. This store lets retrieve_node expand a winning child chunk to its full
parent text. It's a plain JSON file (not in-memory) because the CLI is a fresh process per
question — ingest and query happen in different process runs.
"""
import json

from app import config


def _read_all() -> dict:
    if not config.PARENT_STORE_PATH.exists():
        return {}
    return json.loads(config.PARENT_STORE_PATH.read_text(encoding="utf-8"))


def save_parents(parents: dict) -> None:
    """Merge the given {parent_id: {text, source_doc, section}} entries into the store."""
    config.PARENT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_all()
    existing.update(parents)
    config.PARENT_STORE_PATH.write_text(json.dumps(existing), encoding="utf-8")


def reset_parents() -> None:
    if config.PARENT_STORE_PATH.exists():
        config.PARENT_STORE_PATH.unlink()


def load_parent(parent_id: str) -> dict | None:
    return _read_all().get(parent_id)
