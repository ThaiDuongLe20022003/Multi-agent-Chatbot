"""
Data Retrieval Agent - handles semantic search and information retrieval
"""

import logging
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .base_agent import BaseAgent, AgentContext, AgentResponse
from config.settings import DEFAULT_EMBEDDING_MODEL, PERSIST_DIRECTORY


logger = logging.getLogger(__name__)


class DataRetrievalAgent(BaseAgent):
    """Agent specialized in semantic search and data retrieval"""
    
    def __init__(self):
        super().__init__("data_retriever")
        self.vector_db = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name = DEFAULT_EMBEDDING_MODEL,
            model_kwargs = {'device': 'cpu'},
            encode_kwargs = {'normalize_embeddings': False}
        )
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Retrieve relevant information based on query"""
        try:
            query = context.query
            
            # Check if we have vector DB available
            if not self.vector_db:
                return self._create_error_response("Vector database not initialized")
            
            # Perform similarity search
            relevant_docs = self.vector_db.similarity_search(query, k=4)
            
            # Format results
            retrieved_data = self._format_retrieved_data(relevant_docs)
            
            result_data = {
                "query": query,
                "retrieved_documents": retrieved_data,
                "total_results": len(retrieved_data),
                "search_type": "semantic_similarity"
            }
            
            logger.info(f"Data retrieval completed: {len(retrieved_data)} documents found")
            return self._create_success_response(result_data)
            
        except Exception as e:
            logger.error(f"Data retrieval error: {e}")
            return self._create_error_response(f"Data retrieval failed: {str(e)}")
    
    def initialize_vector_db(self, chunks: List[Dict[str, Any]], collection_name: str = "default"):
        """Initialize vector database with PDF chunks"""
        try:
            from langchain.schema import Document
            
            # Convert chunks to LangChain Documents
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content = chunk["content"],
                    metadata={
                        "page_number": chunk.get("page_number", 1),
                        "chunk_id": chunk.get("chunk_id", ""),
                        **chunk.get("metadata", {})
                    }
                )
                documents.append(doc)
            
            # Create vector store
            self.vector_db = Chroma.from_documents(
                documents = documents,
                embedding = self.embeddings,
                persist_directory = PERSIST_DIRECTORY,
                collection_name = collection_name
            )
            
            logger.info(f"Vector DB initialized with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Vector DB initialization error: {e}")
            return False
    
    def _format_retrieved_data(self, documents: List) -> List[Dict[str, Any]]:
        """Format retrieved documents for response"""
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            formatted_doc = {
                "rank": i + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "page_number": doc.metadata.get("page_number", "unknown"),
                "similarity_score": getattr(doc, 'score', 0.0),
                "metadata": doc.metadata
            }
            formatted_docs.append(formatted_doc)
        
        return formatted_docs
    
    def clear_vector_db(self):
        """Clear the current vector database"""
        if self.vector_db:
            self.vector_db.delete_collection()
            self.vector_db = None
            logger.info("Vector database cleared")