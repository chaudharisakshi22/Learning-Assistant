"""
modules/vector_store.py
───────────────────────
Embeds documents locally using sentence-transformers (no API needed)
and stores them in a local FAISS index.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_ID = os.getenv(
    "EMBEDDING_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2"
)
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", "data/vectorstore")


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_ID,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vectorstore(documents: List[Document]) -> FAISS:
    embeddings = _get_embeddings()
    db = FAISS.from_documents(documents, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    db.save_local(VECTORSTORE_PATH)
    return db


def load_vectorstore() -> Optional[FAISS]:
    index_file = os.path.join(VECTORSTORE_PATH, "index.faiss")
    if not os.path.exists(index_file):
        return None
    embeddings = _get_embeddings()
    return FAISS.load_local(
        VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True
    )


def add_documents(db: FAISS, documents: List[Document]) -> FAISS:
    db.add_documents(documents)
    db.save_local(VECTORSTORE_PATH)
    return db


def get_retriever(db: FAISS, k: int = 4):
    return db.as_retriever(search_kwargs={"k": k})
