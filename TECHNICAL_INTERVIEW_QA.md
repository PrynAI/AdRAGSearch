# Technical Interview Q&A for AdRAGSearch

This document is a project-specific interview preparation guide for discussing AdRAGSearch in AI Engineer, Applied AI, LLM Engineer, and ML Platform interviews.

It is based on the current implementation in this repository:

- `streamlit_app.py`
- `src/document_ingestion/document_processor.py`
- `src/vectorstore/vectorstore.py`
- `src/graph_builder/graph_builder.py`
- `src/nodes/reactnode.py`
- `src/state/rag_state.py`

## 1. Project Overview

### Q1. What is AdRAGSearch?

**Answer:**  
AdRAGSearch is a lightweight single-agent, tool-augmented RAG application. It ingests content from configured web pages and local PDFs, chunks and embeds that content into a FAISS vector store, retrieves relevant chunks for a user query, and then uses a tool-enabled agent to generate the final answer. The UI is built with Streamlit.

### Q2. What problem does this project solve?

**Answer:**  
It solves the problem of asking grounded questions over a mixed knowledge base. Instead of relying only on an LLM’s parametric memory, it retrieves relevant source material first and then generates the answer. It also allows the answering agent to use Wikipedia when general background knowledge is needed beyond the indexed corpus.

### Q3. How would you describe this system in one sentence in an interview?

**Answer:**  
This is a single-agent, tool-augmented RAG system built with LangGraph, FAISS, Streamlit, and OpenAI for querying mixed document sources with source visibility.

### Q4. Is this project basic RAG, agentic RAG, or multi-agent?

**Answer:**  
It is single-agent agentic RAG. It is not just basic retrieve-then-answer, because the responder uses tools. It is not multi-agent, because there is only one tool-using answer agent in the active flow.

## 2. End-to-End Architecture

### Q5. What is the end-to-end flow for a user question?

**Answer:**  
The app initializes by loading source documents, splitting them into chunks, embedding them, and storing them in FAISS. When a user asks a question, the LangGraph workflow first retrieves relevant indexed chunks, then sends the question to the responder node. The responder uses a single agent with tool access to the retriever and Wikipedia, generates an answer, and returns both the answer and source information to the Streamlit UI.

### Q6. What are the main modules in this repo and why are they separated?

**Answer:**  
The repo is split into logical layers:

- `document_ingestion` handles source loading and chunking
- `vectorstore` handles embeddings and retrieval
- `graph_builder` defines workflow orchestration
- `nodes` contains the execution logic for retrieval and answer generation
- `state` defines the shared workflow state
- `streamlit_app.py` handles the UI

This separation makes the code easier to reason about, test, and extend.

### Q7. Why use LangGraph here instead of calling the retriever and LLM directly?

**Answer:**  
LangGraph gives a clear workflow abstraction. Even though the current flow is small, it establishes a structured execution model that can be extended later with query rewriting, routing, grading, retries, or human review. It makes the system more maintainable than putting all logic directly in the UI layer.

### Q8. Why is the graph only two nodes right now?

**Answer:**  
The current MVP only needs a retrieval step and a response-generation step. That keeps the system simple while still leaving a clean path for future graph expansion.

## 3. Document Ingestion

### Q9. What source types does the project support?

**Answer:**  
It supports:

- web URLs through `WebBaseLoader`
- directories of PDFs through `PyPDFDirectoryLoader`
- individual PDF files through `PyMuPDFLoader`
- text files through `TextLoader`

### Q10. Why use `RecursiveCharacterTextSplitter`?

**Answer:**  
It is a practical default for chunking long documents into smaller segments while preserving enough local context for retrieval. It works reasonably well across mixed text sources without requiring domain-specific chunking logic.

### Q11. What is the purpose of chunk size and chunk overlap?

**Answer:**  
Chunk size controls how much content is embedded together. If chunks are too small, context gets fragmented. If chunks are too large, retrieval gets noisy and embeddings become less specific. Overlap helps preserve continuity across chunk boundaries so important information is less likely to be split apart.

### Q12. What chunk settings are used here, and what are the tradeoffs?

**Answer:**  
The current config uses:

- `CHUNK_SIZE = 500`
- `CHUNK_OVERLAP = 50`

This is a reasonable MVP setting for balancing retrieval granularity and context preservation, but it should be tuned per corpus and use case.

### Q13. What are common ingestion issues with web pages and PDFs?

**Answer:**  
Web pages can include navigation, repeated headers, ads, and noisy formatting. PDFs can have broken layout ordering, missing titles, inconsistent metadata, and page extraction problems. Those issues affect retrieval quality because bad chunks lead to bad embeddings.

### Q14. Why are URLs loaded into the vector index at startup instead of searched live?

**Answer:**  
Because this project is designed around indexed retrieval, not live web search. Startup ingestion makes answers grounded in a stable local corpus and keeps runtime query logic simple. The tradeoff is that the content can become stale unless the index is rebuilt.

## 4. Embeddings and Retrieval

### Q15. What embedding approach does the project use?

**Answer:**  
It uses `OpenAIEmbeddings` to convert chunks into dense vectors for semantic search.

### Q16. Why use FAISS?

**Answer:**  
FAISS is a fast and well-known vector search library for dense retrieval. It is a good fit for an MVP because it is lightweight, simple to use locally, and integrates easily with LangChain.

### Q17. How does retrieval work in this project?

**Answer:**  
After document chunks are embedded and stored in FAISS, the vector store is exposed as a retriever. At query time, the retriever returns the most semantically relevant chunks for the user question.

### Q18. What are the current retrieval limitations in this repo?

**Answer:**  
The current retrieval layer is basic:

- no metadata filtering
- no hybrid lexical plus semantic retrieval
- no reranking stage
- no persisted vector index
- no configurable top-k surfaced in the UI

It is enough for an MVP but not optimized for production retrieval quality.

### Q19. How would you improve retrieval quality?

**Answer:**  
I would consider:

- better chunking strategies
- metadata normalization
- hybrid retrieval combining keyword and vector search
- a reranker on top of initial recall
- evaluation with labeled Q&A pairs
- corpus-specific tuning of chunk size, overlap, and retrieval count

## 5. Agent and Tooling Design

### Q20. What makes the responder “agentic”?

**Answer:**  
The responder is not just a direct LLM call. It is created as a tool-using agent that can choose whether to call the retriever tool or the Wikipedia tool before forming its final answer.

### Q21. Why is this still a single-agent system?

**Answer:**  
Because there is only one tool-using agent in the active answer path. There is no planner agent, critic agent, router agent, or separate cooperating agents with distinct roles.

### Q22. What tools does the agent have access to?

**Answer:**  
It currently has two tools:

- `retriever` for the indexed local corpus
- `wikipedia` for general knowledge lookups

### Q23. Why include Wikipedia as a tool?

**Answer:**  
Wikipedia is useful for general background information when the indexed corpus is too narrow to answer a question completely. It extends the system beyond the local documents without adding a full web search integration.

### Q24. What is the difference between the retriever tool and the Wikipedia tool?

**Answer:**  
The retriever tool searches the local indexed corpus, which includes the configured URLs and PDFs already loaded into FAISS. The Wikipedia tool performs an external lookup against Wikipedia only. It is not general internet search.

### Q25. Why is Wikipedia not the same as internet search?

**Answer:**  
Wikipedia is one curated source. Internet search would mean querying the broader live web, including blogs, docs, product pages, and news sources. This repo does not currently have a general live web search tool.

### Q26. How are external sources surfaced in the current code?

**Answer:**  
The responder now captures structured `external_sources` during Wikipedia tool use. Those records contain fields like source type, title, URL, snippet, and query, and the Streamlit UI renders them in the `Sources Used` expander.

### Q27. Why was explicit source capture needed?

**Answer:**  
Without explicit source capture, the UI only knew about the initially retrieved local chunks. Tool outputs from the answering agent were not automatically exposed in a structured way, so external references would be lost from the user-facing source panel.

### Q28. What are the risks of tool-based fallback to Wikipedia?

**Answer:**  
The main risks are:

- the agent may rely on external general knowledge when the indexed corpus should have been prioritized
- Wikipedia is not always the best source for technical or current topics
- there is no trust-ranking or source-quality scoring
- tool latency can increase answer time

## 6. State and Workflow

### Q29. What is stored in `RAGState`?

**Answer:**  
The state currently holds:

- `question`
- `retrieved_docs`
- `external_sources`
- `answer`

This allows the workflow to preserve both local retrieval results and external references through the graph.

### Q30. Why is explicit state useful in LangGraph workflows?

**Answer:**  
Explicit state makes data flow visible and predictable. Instead of passing opaque values between functions, each node updates a defined structure. That helps when you later add more nodes or need to debug intermediate steps.

### Q31. What future nodes could be added to this graph?

**Answer:**  
Useful additions could include:

- a query rewriting node
- a routing node for source selection
- a retrieval grading node
- an answer verification node
- a fallback or retry node

## 7. Streamlit UI and App Behavior

### Q32. Why was Streamlit used?

**Answer:**  
Streamlit is a fast way to turn an AI workflow into a working UI. It is especially useful for demos, MVPs, and internal tools because it requires little frontend code while still making the system interactive.

### Q33. How is expensive initialization handled?

**Answer:**  
The app uses `@st.cache_resource` around the RAG initialization function so the vector store and graph are not rebuilt on every user interaction.

### Q34. What does the UI show to the user?

**Answer:**  
The UI shows:

- the final answer
- indexed document chunks used in retrieval
- external references captured during tool use
- recent search history

### Q35. Why keep recent search history in session state?

**Answer:**  
It improves the user experience in a simple way and avoids the need for a database. It is enough for a single-session demo application.

## 8. Design Tradeoffs

### Q36. What are the main strengths of this implementation?

**Answer:**  
Its strengths are simplicity, modularity, source grounding, and clarity of workflow. It is easy to explain in an interview because the retrieval, agent, workflow, and UI layers are cleanly separated.

### Q37. What are the main limitations of this implementation?

**Answer:**  
The main limitations are:

- no persisted index
- no automated tests
- no evaluation pipeline
- no live internet search
- limited source governance
- basic retrieval quality controls
- no authentication or production observability

### Q38. Why is the vector store rebuilt at startup, and what is the downside?

**Answer:**  
Rebuilding at startup keeps the implementation simple and avoids persistence concerns. The downside is slower startup, repeated embedding cost, and no durable index between sessions.

### Q39. Why not let the app ingest user-uploaded files yet?

**Answer:**  
The current repo is optimized for a controlled MVP setup with configured sources. Upload support adds more UI, validation, storage, and ingestion lifecycle concerns that are outside the current scope.

## 9. Productionization Questions

### Q40. What would you change first to make this production-ready?

**Answer:**  
My first priorities would be:

- persist the vector index
- add ingestion/versioning workflows
- add evaluation and test coverage
- improve retrieval quality
- add structured logging and observability
- add proper secrets handling and deployment setup

### Q41. How would you add live internet search responsibly?

**Answer:**  
I would add a separate web-search tool instead of overloading the Wikipedia tool. I would capture source URLs, snippets, and timestamps in structured form, clearly separate indexed-corpus evidence from live-web evidence in the UI, and add policies so the retriever remains the preferred source for user-provided documents.

### Q42. How would you evaluate this system?

**Answer:**  
I would evaluate it across:

- retrieval quality: recall, precision, and relevance of returned chunks
- answer quality: groundedness, completeness, and factual correctness
- source attribution quality: whether surfaced sources actually support the answer
- latency and cost
- behavior under out-of-domain or low-context questions

### Q43. How would you test this repo?

**Answer:**  
I would add:

- unit tests for document loading and chunking
- tests for vector store creation and retrieval behavior
- tests for agent tool wrappers
- tests that verify `external_sources` are returned when Wikipedia is used
- UI-level smoke tests for rendering source sections

### Q44. What security or privacy issues matter here?

**Answer:**  
Important considerations include:

- sensitive documents being embedded into third-party services
- API key handling
- data retention policy for embedded content
- preventing prompt injection from untrusted web content
- safe handling of external tool outputs

## 10. Interview Follow-Up Questions

### Q45. If an interviewer asks, “Why not just call the LLM directly?”, what should you say?

**Answer:**  
A direct LLM call would rely mostly on parametric memory and would be less grounded. RAG gives the model access to relevant source material, and the tool-enabled agent gives it controlled access to multiple information paths.

### Q46. If an interviewer asks, “Why didn’t you build multi-agent?”, what should you say?

**Answer:**  
Because multi-agent orchestration would add complexity without strong evidence it is needed for this MVP. A single-agent design is easier to reason about, cheaper to run, and sufficient for the current use case of retrieval plus fallback knowledge lookup.

### Q47. If an interviewer asks, “What is the most important engineering lesson from this project?”, what should you say?

**Answer:**  
The most important lesson is that source handling and state design matter as much as answer generation. If you do not preserve retrieval outputs and external tool evidence in a structured way, the system becomes harder to debug, trust, and improve.

### Q48. If an interviewer asks, “What would you improve next?”, what should you say?

**Answer:**  
I would improve retrieval quality, persist the vector index, add evaluation and tests, and introduce a clearly separated live web-search tool if the product needed broader or fresher information than Wikipedia provides.

## 11. Rapid Interview Pitch

### Q49. Give me a 30-second technical summary of this project.

**Answer:**  
I built a single-agent, tool-augmented RAG system using LangGraph, LangChain, FAISS, Streamlit, and OpenAI. It ingests configured web pages and local PDFs, chunks and embeds them into a vector index, retrieves relevant context for a user query, and uses a tool-enabled responder agent to answer with access to both the local corpus and Wikipedia. I also made the UI surface both indexed sources and external references used during answer generation.

### Q50. Give me a strong closing line for interviews.

**Answer:**  
This project demonstrates that I can take an LLM workflow from ingestion and retrieval through orchestration, tool use, source attribution, and UI delivery, while still reasoning clearly about tradeoffs and next-step production concerns.
