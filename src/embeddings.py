from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Returns a HuggingFaceEmbeddings model."""
    return HuggingFaceEmbeddings(model_name=model_name)
