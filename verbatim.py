from __future__ import annotations
from pathlib import Path
from typing import List, Any

DEFAULT_DOCS_DIR = Path("/Users/tuvshinselenge/Documents/Documents - MacBook Air von Tuvshin/TU/Verbatim_RAG/doc").resolve()
DEFAULT_DB_PATH  = Path("/Users/tuvshinselenge/Documents/Documents - MacBook Air von Tuvshin/TU/Verbatim_RAG/index.db").resolve()

SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from verbatim_rag import VerbatimIndex
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.extractors import ModelSpanExtractor
from verbatim_rag.ingestion import DocumentProcessor


def _process_pdf(proc: DocumentProcessor, pdf_path: Path):
    p = str(pdf_path)
    try:
        return proc.process_file(p, title=pdf_path.name, metadata={"source": p})
    except TypeError:
        try:
            return proc.process_file(p)
        except AttributeError:
            return proc.process_path(p)


def build_index(docs_dir: str | Path = DEFAULT_DOCS_DIR,
                db_path: str | Path = DEFAULT_DB_PATH) -> str:

    docs_dir = Path(docs_dir).expanduser().resolve()
    db_path = Path(db_path).expanduser().resolve()

    pdfs: List[Path] = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")

    proc = DocumentProcessor()
    docs = []
    for pdf in pdfs:
        d = _process_pdf(proc, pdf)
        docs.append(d)

    index = VerbatimIndex(sparse_model=SPARSE_MODEL, db_path=str(db_path))
    index.add_documents(docs)
    return str(db_path)


def _index(db_path: str | Path) -> VerbatimIndex:
    return VerbatimIndex(sparse_model=SPARSE_MODEL, db_path=str(Path(db_path)))


def ask(question: str,
        db_path: str | Path = DEFAULT_DB_PATH,
        k: int = 16):

    if not question or not str(question).strip():
        raise ValueError("Question is empty.")
    idx = _index(db_path)
    rag = VerbatimRAG(index=idx, k=k)
    return rag.query(str(question).strip())


def format_answer(resp: Any) -> str:
    if resp is None:
        return "No answer. Check that your index contains documents."

    parts: List[str] = []
    ans = (getattr(resp, "answer", "") or "").strip()
    if ans:
        parts.append(ans)

    cites = getattr(resp, "citations", []) or []
    if cites:
        parts.append("\nSources:")
        for i, c in enumerate(cites, 1):
            try:
                text = getattr(c, "text", None) or (c.get("text") if isinstance(c, dict) else "")
                src  = getattr(c, "source", None) or (c.get("source") if isinstance(c, dict) else "")
                page = getattr(c, "page", None) or (c.get("page") if isinstance(c, dict) else None)
                line = f"{i}. {text}"
                meta = []
                if src:  meta.append(f"Source: {src}")
                if page is not None: meta.append(f"Page: {page}")
                if meta: line += "  (" + " | ".join(meta) + ")"
            except Exception:
                line = f"{i}. {c}"
            parts.append(line)

    out = "\n".join([p for p in parts if p]).strip()
    return out or "No evidence found in the PDF."


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Build index and ask questions (verbatim-rag).")
    p.add_argument("--build", action="store_true", help="Build/refresh the index from PDFs in docs dir.")
    p.add_argument("--docs", default=str(DEFAULT_DOCS_DIR), help="Folder with PDFs.")
    p.add_argument("--db",   default=str(DEFAULT_DB_PATH),  help="SQLite index path (index.db).")
    p.add_argument("question", nargs="*", help="Question to ask")
    args = p.parse_args()

    if args.build:
        out = build_index(args.docs, args.db)
        print(f"Index ready at: {out}")

    if args.question:
        q = " ".join(args.question).strip()
        resp = ask(q, args.db)
        print(format_answer(resp))
