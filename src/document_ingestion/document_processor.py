"""Document loading and splitting utilities."""

from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handle document loading and chunking."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Initialize the document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url(self, url: str) -> List[Document]:
        """Load documents from a URL."""
        loader = WebBaseLoader(url)
        return loader.load()

    def load_from_pdf_dir(self, directory: Union[str, Path]) -> List[Document]:
        """Load documents from all PDFs inside a directory."""
        loader = PyPDFDirectoryLoader(str(directory))
        return loader.load()

    def load_from_pdf(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a single PDF file."""
        loader = PyMuPDFLoader(str(file_path))
        return loader.load()

    def load_from_txt(self, file_path: Union[str, Path]) -> List[Document]:
        """Load documents from a text file."""
        loader = TextLoader(str(file_path), encoding="utf-8")
        return loader.load()

    def load_source(self, source: Union[str, Path]) -> List[Document]:
        """Load documents from a single source."""
        source_str = str(source)
        if source_str.startswith(("http://", "https://")):
            return self.load_from_url(source_str)

        path = Path(source)
        if path.is_dir():
            return self.load_from_pdf_dir(path)
        if path.is_file() and path.suffix.lower() == ".pdf":
            return self.load_from_pdf(path)
        if path.is_file() and path.suffix.lower() == ".txt":
            return self.load_from_txt(path)

        raise ValueError(
            f"Unsupported source: {source_str}. Expected a URL, .pdf file, .txt file, or directory."
        )

    def load_documents(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Load documents from multiple sources."""
        docs: List[Document] = []
        for source in sources:
            docs.extend(self.load_source(source))
        return docs

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        return self.splitter.split_documents(documents)

    def process_sources(self, sources: List[Union[str, Path]]) -> List[Document]:
        """Load and split documents from mixed sources."""
        docs = self.load_documents(sources)
        return self.split_documents(docs)
