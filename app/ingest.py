"""Load policy docs, split into hierarchical (parent section / child chunk) pieces, and persist
child-chunk embeddings to Chroma plus parent text to the parent store.

Run with: python -m app.ingest (rebuilds the whole index from data/policies/).
Also exposes ingest_file() for adding a single new document incrementally (used by the
Streamlit "upload a document" feature) without re-embedding everything else.
"""
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config, parent_store
from app.vectorstore import get_embeddings, get_store, reset_store

SECTION_RE = re.compile(r"^##\s+(§\d+.*)$", re.MULTILINE)
MARKDOWN_HEADER_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


def extract_text(path: Path) -> str:
    """Read a document's text content, dispatching on file extension."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def _doc_title(text: str, fallback: str) -> str:
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return title_match.group(1).strip() if title_match else fallback


def _sections_from_matches(text: str, matches: list[re.Match], source_doc: str) -> list[Document]:
    docs = []
    for i, match in enumerate(matches):
        section_title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        docs.append(
            Document(
                page_content=f"{section_title}\n{section_text}",
                metadata={"source_doc": source_doc, "section": section_title},
            )
        )
    return docs


def _pseudo_sections(text: str, source_doc: str) -> list[Document]:
    """Last-resort fallback for unstructured text: fixed-size, labeled 'Part N' sections."""
    docs = []
    for i in range(0, len(text), config.PARENT_MAX_CHARS):
        part_text = text[i : i + config.PARENT_MAX_CHARS].strip()
        if not part_text:
            continue
        section_title = f"Part {len(docs) + 1}"
        docs.append(
            Document(
                page_content=f"{section_title}\n{part_text}",
                metadata={"source_doc": source_doc, "section": section_title},
            )
        )
    return docs


def _cap_section_size(doc: Document) -> list[Document]:
    """Split an oversized section into multiple parent-sized pieces so none exceed PARENT_MAX_CHARS."""
    text = doc.page_content
    if len(text) <= config.PARENT_MAX_CHARS:
        return [doc]
    base_section = doc.metadata["section"]
    pieces = []
    for i in range(0, len(text), config.PARENT_MAX_CHARS):
        piece_text = text[i : i + config.PARENT_MAX_CHARS].strip()
        if not piece_text:
            continue
        part_num = len(pieces) + 1
        metadata = dict(doc.metadata)
        metadata["section"] = base_section if part_num == 1 else f"{base_section} (part {part_num})"
        pieces.append(Document(page_content=piece_text, metadata=metadata))
    return pieces


def _split_into_sections(text: str, source_doc: str) -> list[Document]:
    """Split a document into parent sections, trying progressively looser conventions:

    1. '§N' headers (the policy doc convention) — gives precise, citable sections.
    2. Generic markdown headers ('#'/'##'/'###') — for structured uploads that use normal headings.
    3. Fixed-size pseudo-sections — for fully unstructured text (e.g. a plain PDF with no headings).

    Every resulting section is then capped at PARENT_MAX_CHARS so no single parent (even a real
    §N or markdown section) can blow up the generation context for a long document.
    """
    section_matches = list(SECTION_RE.finditer(text))
    if section_matches:
        sections = _sections_from_matches(text, section_matches, source_doc)
    else:
        header_matches = list(MARKDOWN_HEADER_RE.finditer(text))
        if header_matches:
            sections = _sections_from_matches(text, header_matches, source_doc)
        else:
            sections = _pseudo_sections(text, source_doc) or [
                Document(page_content=text, metadata={"source_doc": source_doc, "section": "Full Document"})
            ]

    capped = []
    for section in sections:
        capped.extend(_cap_section_size(section))
    return capped


def load_section_documents() -> list[Document]:
    docs = []
    for path in sorted(config.DATA_DIR.glob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        text = extract_text(path)
        doc_title = _doc_title(text, path.stem)
        docs.extend(_split_into_sections(text, doc_title))
    return docs


def chunk_documents(section_docs: list[Document]) -> tuple[list[Document], dict]:
    """Split each parent section into small child chunks for embedding/retrieval.

    Returns (child_chunks, parents) where parents maps parent_id -> {text, source_doc, section},
    to be persisted via app.parent_store so retrieval can expand a matched child chunk back to
    its full parent section text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = []
    parents = {}
    for doc in section_docs:
        parent_id = f"{doc.metadata['source_doc']}::{doc.metadata['section']}"
        parents[parent_id] = {
            "text": doc.page_content,
            "source_doc": doc.metadata["source_doc"],
            "section": doc.metadata["section"],
        }
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = f"{parent_id}::{i}"
            metadata["parent_id"] = parent_id
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks, parents


def build_index() -> int:
    """Wipe and rebuild the whole index (child chunks + parent store) from data/policies/."""
    from langchain_chroma import Chroma

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    reset_store()
    parent_store.reset_parents()

    chunks, parents = chunk_documents(load_section_documents())

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    store.add_documents(chunks)
    parent_store.save_parents(parents)
    return len(chunks)


def ingest_file(path: Path) -> int:
    """Add a single document to the existing index without rebuilding everything else.

    Used by the Streamlit upload feature: the file should already live under data/policies/.
    """
    text = extract_text(path)
    doc_title = _doc_title(text, path.stem)
    chunks, parents = chunk_documents(_split_into_sections(text, doc_title))
    if chunks:
        get_store().add_documents(chunks)
        parent_store.save_parents(parents)
    return len(chunks)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks from {config.DATA_DIR} into {config.CHROMA_DIR}")
