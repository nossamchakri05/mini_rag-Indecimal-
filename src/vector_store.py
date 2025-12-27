import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

def create_vector_store(documents: List[Document], embeddings: Embeddings) -> FAISS:
    """Creates a FAISS vector store from documents."""
    return FAISS.from_documents(documents, embeddings)

def save_vector_store(vector_store: FAISS, path: str):
    """Saves the vector store to disk."""
    vector_store.save_local(path)

def load_vector_store(path: str, embeddings: Embeddings) -> FAISS:
    """Loads the vector store from disk."""
    if os.path.exists(path):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    return None
