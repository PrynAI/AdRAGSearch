"""RAG state definition for the LangGraph workflow."""

from typing import List

from langchain_core.documents import Document
from pydantic import BaseModel, Field


class RAGState(BaseModel):
    """State object for the RAG workflow."""

    question: str
    retrieved_docs: List[Document] = Field(default_factory=list)
    external_sources: List[dict[str, str]] = Field(default_factory=list)
    answer: str = ""
