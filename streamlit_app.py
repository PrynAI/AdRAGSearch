"""Streamlit UI for Agentic RAG System - Simplified Version"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorstore.vectorstore import VectorStore
from src.graph_builder.graph_builder import GraphBuilder

# Page configuration
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []

def _source_name(source: str) -> str:
    """Extract a readable name from a file path or URL."""
    if not source:
        return "Unknown source"
    return source.replace("\\", "/").rstrip("/").split("/")[-1] or source

def _render_indexed_documents(documents):
    """Render indexed document chunks with source metadata."""
    st.markdown("#### Indexed Documents")
    for i, doc in enumerate(documents, 1):
        metadata = getattr(doc, "metadata", {}) or {}
        source = str(metadata.get("source", "")).strip()
        title = metadata.get("title") or _source_name(source) or f"Document {i}"
        page_label = metadata.get("page_label")
        page = metadata.get("page")

        st.markdown(f"**{i}. {title}**")
        if source.startswith(("http://", "https://")):
            st.markdown(f"[Open source]({source})")
        elif source:
            st.caption(f"Source: {source}")

        if page_label:
            st.caption(f"Page: {page_label}")
        elif isinstance(page, int):
            st.caption(f"Page: {page + 1}")

        preview = doc.page_content.strip()
        if len(preview) > 400:
            preview = preview[:400].rstrip() + "..."
        st.text_area(
            f"Indexed Document {i}",
            preview,
            height=120,
            disabled=True,
            key=f"indexed_doc_{i}_{source}_{page_label or page or 'na'}",
        )

def _render_external_sources(external_sources):
    """Render external references collected during tool use."""
    st.markdown("#### External References")
    for i, source in enumerate(external_sources, 1):
        source_type = source.get("source_type", "external").title()
        title = source.get("title") or f"External Source {i}"
        url = source.get("url", "")
        query = source.get("query", "")
        snippet = source.get("snippet", "")

        st.markdown(f"**{i}. {title}**")
        if query:
            st.caption(f"{source_type} lookup for: {query}")
        else:
            st.caption(source_type)

        if url:
            st.markdown(f"[Open source]({url})")

        if snippet:
            st.text_area(
                f"External Reference {i}",
                snippet,
                height=120,
                disabled=True,
                key=f"external_ref_{i}_{title}_{url}",
            )

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    try:
        # Initialize components
        llm = Config.get_llm()
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        vector_store = VectorStore()
        
        # Load both configured URLs and local PDFs.
        sources = [*Config.DEFAULT_URLS, Config.DEFAULT_PDF_DIR]

        # Process documents
        documents = doc_processor.process_sources(sources)
        
        # Create vector store
        vector_store.create_vectorstore(documents)
        
        # Build graph
        graph_builder = GraphBuilder(
            retriever=vector_store.get_retriever(),
            llm=llm
        )
        graph_builder.build()
        
        return graph_builder, len(documents)
    except Exception as e:
        st.error(f"Failed to initialize: {str(e)}")
        return None, 0

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"✅ System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know?"
        )
        submit = st.form_submit_button("🔍 Search")
    
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                
                # Add to history
                st.session_state.history.append({
                    'question': question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.success(result['answer'])
                
                # Show local and external sources in expander
                retrieved_docs = result.get('retrieved_docs', [])
                external_sources = result.get('external_sources', [])
                with st.expander("📄 Sources Used"):
                    if retrieved_docs:
                        _render_indexed_documents(retrieved_docs)

                    if external_sources:
                        _render_external_sources(external_sources)

                    if not retrieved_docs and not external_sources:
                        st.info("No source details were captured for this answer.")
                
                st.caption(f"⏱️ Response time: {elapsed_time:.2f} seconds")
    
    # Show history
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):  # Show last 3
            with st.container():
                st.markdown(f"**Q:** {item['question']}")
                st.markdown(f"**A:** {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")

if __name__ == "__main__":
    main()
