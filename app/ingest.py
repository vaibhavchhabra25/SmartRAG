"""Load policy docs, split into cited chunks, and persist embeddings to Chroma.

Run with: python -m app.ingest (rebuilds the whole index from data/policies/).
Also exposes ingest_file() for adding a single new document incrementally (used by the
Streamlit "upload a document" feature) without re-embedding everything else.
"""
import re
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app import config
from app.vectorstore import get_embeddings, get_store, reset_store

SECTION_RE = re.compile(r"^##\s+(§\d+.*)$", re.MULTILINE)


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


def _split_into_sections(text: str, source_doc: str) -> list[Document]:
    """Split a policy doc into (section_title, section_text) Documents using its §N headers.

    Docs without §N headers (e.g. an uploaded PDF with no such convention) fall back to a
    single "Full Document" section.
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [Document(page_content=text, metadata={"source_doc": source_doc, "section": "Full Document"})]

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


def load_section_documents() -> list[Document]:
    docs = []
    for path in sorted(config.DATA_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_title = _doc_title(text, path.stem)
        docs.extend(_split_into_sections(text, doc_title))
    return docs


def chunk_documents(section_docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    chunks = []
    for doc in section_docs:
        for i, piece in enumerate(splitter.split_text(doc.page_content)):
            metadata = dict(doc.metadata)
            metadata["chunk_id"] = f"{metadata['source_doc']}::{metadata['section']}::{i}"
            chunks.append(Document(page_content=piece, metadata=metadata))
    return chunks


def build_index() -> int:
    """Wipe and rebuild the whole index from every file in data/policies/."""
    from langchain_chroma import Chroma

    if config.CHROMA_DIR.exists():
        shutil.rmtree(config.CHROMA_DIR)
    reset_store()

    chunks = chunk_documents(load_section_documents())

    store = Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(config.CHROMA_DIR),
    )
    store.add_documents(chunks)
    return len(chunks)


def ingest_file(path: Path) -> int:
    """Add a single document to the existing index without rebuilding everything else.

    Used by the Streamlit upload feature: the file should already live under data/policies/.
    """
    text = extract_text(path)
    doc_title = _doc_title(text, path.stem)
    chunks = chunk_documents(_split_into_sections(text, doc_title))
    if chunks:
        get_store().add_documents(chunks)
    return len(chunks)


if __name__ == "__main__":
    count = build_index()
    print(f"Indexed {count} chunks from {config.DATA_DIR} into {config.CHROMA_DIR}")
