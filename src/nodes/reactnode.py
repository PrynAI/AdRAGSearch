"""LangGraph nodes for the RAG workflow and answer agent."""

from contextvars import ContextVar
from typing import List, Optional

import wikipedia
from langchain.agents import create_agent
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool

from src.state.rag_state import RAGState


class RAGNodes:
    """Contains node functions for the RAG workflow."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None
        self._external_source_collector: ContextVar[Optional[list[dict[str, str]]]] = (
            ContextVar("external_source_collector", default=None)
        )

    @staticmethod
    def _trim_text(text: str, limit: int = 900) -> str:
        """Collapse whitespace and trim long tool outputs for readability."""
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _message_to_text(content) -> str:
        """Normalize LangChain message content into plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        chunks.append(text)
                elif item is not None:
                    chunks.append(str(item))
            return "\n".join(chunks).strip()
        return str(content) if content is not None else ""

    def _record_external_sources(self, sources: list[dict[str, str]]) -> None:
        """Append deduplicated external references for the current agent run."""
        collector = self._external_source_collector.get()
        if collector is None:
            return

        seen = {
            (item.get("source_type", ""), item.get("title", ""), item.get("url", ""))
            for item in collector
        }
        for source in sources:
            key = (
                source.get("source_type", ""),
                source.get("title", ""),
                source.get("url", ""),
            )
            if key in seen:
                continue
            collector.append(source)
            seen.add(key)

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Retrieve indexed documents for the incoming question."""
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question, retrieved_docs=docs)

    def _build_tools(self) -> List[Tool]:
        """Build retriever and Wikipedia tools for the answer agent."""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."

            merged = []
            for i, doc in enumerate(docs[:8], start=1):
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                title = metadata.get("title") or metadata.get("source") or f"doc_{i}"
                source = metadata.get("source")
                prefix = f"[{i}] {title}"
                if source and source != title:
                    prefix = f"{prefix}\nSource: {source}"
                merged.append(f"{prefix}\n{self._trim_text(doc.page_content, limit=1200)}")
            return "\n\n".join(merged)

        def wikipedia_tool_fn(query: str) -> str:
            """Run a Wikipedia lookup and capture source metadata for the UI."""
            wikipedia.set_lang("en")
            page_titles = wikipedia.search(query, results=3)
            if not page_titles:
                return "No Wikipedia results found."

            formatted_results = []
            captured_sources: list[dict[str, str]] = []

            for title in page_titles:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                except wikipedia.exceptions.DisambiguationError as exc:
                    if not exc.options:
                        continue
                    try:
                        page = wikipedia.page(exc.options[0], auto_suggest=False)
                    except (
                        wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.PageError,
                    ):
                        continue
                except wikipedia.exceptions.PageError:
                    continue
                except Exception as exc:
                    return f"Wikipedia lookup failed: {exc}"

                snippet = self._trim_text(page.summary)
                captured_sources.append(
                    {
                        "source_type": "wikipedia",
                        "title": page.title,
                        "url": page.url,
                        "snippet": snippet,
                        "query": query,
                    }
                )
                formatted_results.append(
                    f"[{len(captured_sources)}] {page.title}\n"
                    f"URL: {page.url}\n"
                    f"Summary: {snippet}"
                )

            if not captured_sources:
                return "No Wikipedia results found."

            self._record_external_sources(captured_sources)
            return "\n\n".join(formatted_results)

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn,
        )
        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general knowledge.",
            func=wikipedia_tool_fn,
        )
        return [retriever_tool, wikipedia_tool]

    def _build_agent(self):
        """Create the tool-using answer agent."""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_agent(self.llm, tools=tools, system_prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """Generate an answer and capture any external sources used."""
        if self._agent is None:
            self._build_agent()

        external_sources: list[dict[str, str]] = []
        token = self._external_source_collector.set(external_sources)
        try:
            result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})
        finally:
            self._external_source_collector.reset(token)

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer = self._message_to_text(getattr(messages[-1], "content", None))

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            external_sources=external_sources,
            answer=answer or "Could not generate answer.",
        )



