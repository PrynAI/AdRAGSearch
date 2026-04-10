""" Langgraph nodes for RAG workflow"""

from src.state.rag_state import RAGState

class RAGNodes:
    """ Contianes nodes function for RAG workflow"""

    def __init__(self, retriever, llm):
        """ Initialize RAG nodes
        
        Args:
            retriever:Document retriever instance
            llm:Language model instance
        """
        self.retriever=retriever
        self.llm=llm

    def retriever_docs(self, state:RAGState)->RAGState:
        """
        Retrieve relevant documents node
        Args:
         state:Current RAG state

         Returns:
         Updated RAG state with retrieved documents

        """
        docs=self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )