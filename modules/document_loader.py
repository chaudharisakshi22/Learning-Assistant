"""
modules/document_loader.py
──────────────────────────
Loads PDFs, plain-text files, and web URLs into LangChain Documents,
then splits them into overlapping chunks ready for embedding.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 80

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    pages  = loader.load()
    return _splitter.split_documents(pages)


def load_text(path: str) -> List[Document]:
    loader = TextLoader(path, encoding="utf-8")
    docs   = loader.load()
    return _splitter.split_documents(docs)


def load_url(url: str) -> List[Document]:
    loader = WebBaseLoader(url)
    docs   = loader.load()
    return _splitter.split_documents(docs)


def load_any(source: str) -> List[Document]:
    source = source.strip()
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {source}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(source)
    elif suffix in {".txt", ".md"}:
        return load_text(source)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .pdf, .txt, or a URL.")
