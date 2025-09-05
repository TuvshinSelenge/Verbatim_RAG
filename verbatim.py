from __future__ import annotations
from pathlib import Path
from typing import List, Any
import os

DEFAULT_DOCS_DIR = Path("/Users/tuvshinselenge/Documents/Documents - MacBook Air von Tuvshin/TU/Verbatim_RAG/doc").resolve()
DEFAULT_DB_PATH  = Path("/Users/tuvshinselenge/Documents/Documents - MacBook Air von Tuvshin/TU/Verbatim_RAG/index.db").resolve()

SPARSE_MODEL = "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill"

# Avoid HF tokenizer parallelism warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from verbatim_rag import VerbatimIndex
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.ingestion import DocumentProcessor


def _process_pdf(proc: DocumentProcessor, pdf_path: Path):
    p = str(pdf_path)
    return proc.process_file(p, title=pdf_path.name, metadata={"source": p})


def build_index(docs_dir: str | Path = DEFAULT_DOCS_DIR,
                db_path: str | Path = DEFAULT_DB_PATH) -> str:
    docs_dir = Path(docs_dir).expanduser().resolve()
    db_path  = Path(db_path).expanduser().resolve()

    pdfs: List[Path] = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {docs_dir}")

    proc = DocumentProcessor()
    docs = [_process_pdf(proc, pdf) for pdf in pdfs]

    index = VerbatimIndex(sparse_model=SPARSE_MODEL, db_path=str(db_path))
    index.add_documents(docs)
    return str(db_path)

def _index(db_path: str | Path) -> VerbatimIndex:
    return VerbatimIndex(sparse_model=SPARSE_MODEL, db_path=str(Path(db_path)))

def clean_text(text: str) -> str:
    """Remove unwanted symbols like bullets and extra whitespace."""
    return text.replace("•", "").strip()

def ask(question: str,
        db_path: str | Path = DEFAULT_DB_PATH,
        k: int = 16):
    idx = _index(db_path)
    rag = VerbatimRAG(index=idx, k=k)
    return rag.query(str(question).strip())


def format_answer(resp: Any) -> str:
    if resp is None:
        return "No answer. Check that your index contains documents."
    ans = clean_text(getattr(resp, "answer", "") or "")
    return ans or "No evidence found in the PDF."
