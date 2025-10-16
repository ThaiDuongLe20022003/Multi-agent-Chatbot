"""
Data Retrieval Agent with horizontal collaboration capabilities.
Enhanced for peer-to-peer communication and specialized data provisioning.
"""

import logging
from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from .base_agent import BaseAgent, AgentContext, AgentResponse
from config.settings import DEFAULT_EMBEDDING_MODEL, PERSIST_DIRECTORY


logger = logging.getLogger(__name__)


class DataRetrievalAgent(BaseAgent):
    """Agent specialized in semantic search and data retrieval with horizontal collaboration"""
    
    def __init__(self):
        super().__init__("data_retriever")
        self.vector_db = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Retrieve relevant information with horizontal collaboration"""
        start_time = self._get_timestamp()
        collaborations = []
        
        try:
            query = context.query
            
            # Check if we have vector DB available
            if not self.vector_db:
                return self._create_error_response("Vector database not initialized")
            
            # PERFORM SEMANTIC SEARCH
            relevant_docs = self.vector_db.similarity_search(query, k=4)
            
            # BROADCAST search results to interested peer agents
            broadcast_responses = self.broadcast_message(
                "search_results_available",
                {
                    "query": query,
                    "documents_found": len(relevant_docs),
                    "search_type": "semantic_similarity",
                    "results_sample": [doc.page_content[:100] + "..." for doc in relevant_docs[:2]]
                }
            )
            collaborations.extend(broadcast_responses)
            
            # Format results for horizontal consumption
            retrieved_data = self._format_retrieved_data(relevant_docs)
            
            result_data = {
                "query": query,
                "retrieved_documents": retrieved_data,
                "total_results": len(retrieved_data),
                "search_type": "semantic_similarity",
                "collaboration_announcements": len(broadcast_responses)
            }
            
            processing_time = self._calculate_processing_time(start_time)
            logger.info(f"✅ Data retrieval completed: {len(retrieved_data)} documents, {len(collaborations)} collaborations")
            
            return self._create_success_response(
                data=result_data,
                collaborations=collaborations,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Data retrieval error: {e}")
            return self._create_error_response(f"Data retrieval failed: {str(e)}")
    
    def initialize_vector_db(self, chunks: List[Dict[str, Any]], collection_name: str = "default"):
        """Initialize vector database with PDF chunks and announce availability"""
        try:
            from langchain.schema import Document
            
            # Convert chunks to LangChain Documents
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk["content"],
                    metadata={
                        "page_number": chunk.get("page_number", 1),
                        "chunk_id": chunk.get("chunk_id", ""),
                        **chunk.get("metadata", {})
                    }
                )
                documents.append(doc)
            
            # Create vector store
            self.vector_db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=PERSIST_DIRECTORY,
                collection_name=collection_name
            )
            
            # ANNOUNCE vector DB availability to peer agents
            self.broadcast_message(
                "vector_db_ready",
                {
                    "documents_count": len(documents),
                    "collection_name": collection_name,
                    "embedding_model": DEFAULT_EMBEDDING_MODEL
                }
            )
            
            logger.info(f"✅ Vector DB initialized with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ Vector DB initialization error: {e}")
            return False
    
    def _handle_data_request(self, content: Dict) -> Dict:
        """
        Handle specialized data requests from peer agents in horizontal collaboration
        Example: Legal Agent requests legal-specific documents
        """
        try:
            request_type = content.get("request_type", "general")
            query = content.get("query", "")
            current_docs = content.get("current_docs", [])
            legal_topics = content.get("legal_topics", [])
            
            if not self.vector_db:
                return {"status": "error", "message": "Vector database not available"}
            
            if request_type == "legal_specific":
                # Enhanced search for legal-specific content
                legal_query = self._build_legal_query(query, legal_topics)
                legal_docs = self.vector_db.similarity_search(legal_query, k=6)
                
                # Filter and enhance results for legal analysis
                enhanced_docs = self._enhance_for_legal_analysis(legal_docs, legal_topics)
                
                return {
                    "status": "success",
                    "request_type": "legal_specific",
                    "enhanced_results": enhanced_docs,
                    "original_query": query,
                    "legal_query_used": legal_query,
                    "documents_found": len(enhanced_docs)
                }
            
            elif request_type == "comprehensive":
                # Comprehensive search with multiple strategies
                basic_docs = self.vector_db.similarity_search(query, k=4)
                expanded_query = query + " " + " ".join(content.get("expansion_terms", []))
                expanded_docs = self.vector_db.similarity_search(expanded_query, k=3)
                
                combined_docs = self._merge_and_deduplicate(basic_docs, expanded_docs)
                
                return {
                    "status": "success",
                    "request_type": "comprehensive",
                    "combined_results": self._format_retrieved_data(combined_docs),
                    "search_strategies_used": ["basic_semantic", "expanded_query"],
                    "total_unique_documents": len(combined_docs)
                }
            
            else:
                # General data provision
                general_docs = self.vector_db.similarity_search(query, k=4)
                return {
                    "status": "success",
                    "request_type": "general",
                    "documents": self._format_retrieved_data(general_docs),
                    "search_parameters": {"k": 4, "strategy": "semantic_similarity"}
                }
                
        except Exception as e:
            logger.error(f"Error handling data request: {e}")
            return {"status": "error", "message": f"Data request failed: {str(e)}"}
    
    def _handle_analysis_request(self, content: Dict) -> Dict:
        """
        Handle analysis requests from peer agents
        Example: QA Agent requests retrieval quality analysis
        """
        try:
            analysis_type = content.get("analysis_type", "retrieval_quality")
            retrieved_docs = content.get("retrieved_documents", [])
            original_query = content.get("query", "")
            
            if analysis_type == "retrieval_quality":
                if not retrieved_docs:
                    return {"status": "error", "message": "No documents provided for analysis"}
                
                # Analyze retrieval quality metrics
                quality_metrics = self._analyze_retrieval_quality(retrieved_docs, original_query)
                
                return {
                    "status": "success",
                    "analysis_type": "retrieval_quality",
                    "quality_metrics": quality_metrics,
                    "recommendations": self._generate_retrieval_recommendations(quality_metrics)
                }
            
            elif analysis_type == "coverage_analysis":
                # Analyze document coverage and diversity
                coverage_metrics = self._analyze_document_coverage(retrieved_docs)
                
                return {
                    "status": "success",
                    "analysis_type": "coverage_analysis",
                    "coverage_metrics": coverage_metrics,
                    "diversity_score": self._calculate_diversity_score(retrieved_docs)
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Data retrieval analysis capabilities available",
                    "available_analyses": ["retrieval_quality", "coverage_analysis", "relevance_analysis"]
                }
                
        except Exception as e:
            logger.error(f"Error handling analysis request: {e}")
            return {"status": "error", "message": f"Analysis request failed: {str(e)}"}
    
    def _handle_validation_request(self, content: Dict) -> Dict:
        """
        Handle validation requests from peer agents
        Example: Summarize Agent validates source coverage
        """
        try:
            validation_type = content.get("validation_type", "source_coverage")
            documents_to_validate = content.get("documents", [])
            validation_criteria = content.get("criteria", {})
            
            if validation_type == "source_coverage":
                coverage_validation = self._validate_source_coverage(documents_to_validate, validation_criteria)
                
                return {
                    "status": "success",
                    "validation_type": "source_coverage",
                    "is_sufficient": coverage_validation["is_sufficient"],
                    "coverage_score": coverage_validation["coverage_score"],
                    "missing_aspects": coverage_validation["missing_aspects"],
                    "recommendations": coverage_validation["recommendations"]
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Data validation capabilities available",
                    "available_validations": ["source_coverage", "relevance_validation", "diversity_validation"]
                }
                
        except Exception as e:
            logger.error(f"Error handling validation request: {e}")
            return {"status": "error", "message": f"Validation request failed: {str(e)}"}
    
    def _build_legal_query(self, original_query: str, legal_topics: List[str]) -> str:
        """Build enhanced query for legal-specific retrieval"""
        base_query = original_query
        if legal_topics:
            topic_enhancement = " " + " ".join([f"legal {topic}" for topic in legal_topics])
            return base_query + topic_enhancement
        return base_query
    
    def _enhance_for_legal_analysis(self, docs: List, legal_topics: List[str]) -> List[Dict]:
        """Enhance documents for legal analysis with relevance scoring"""
        enhanced_docs = []
        for doc in docs:
            content = doc.page_content.lower()
            relevance_score = sum(1 for topic in legal_topics if topic.replace('_', ' ') in content)
            
            enhanced_doc = {
                "content": doc.page_content,
                "page_number": doc.metadata.get("page_number", "unknown"),
                "relevance_score": relevance_score,
                "legal_topics_matched": [topic for topic in legal_topics if topic.replace('_', ' ') in content],
                "metadata": doc.metadata
            }
            enhanced_docs.append(enhanced_doc)
        
        # Sort by relevance score
        enhanced_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return enhanced_docs
    
    def _format_retrieved_data(self, documents: List) -> List[Dict[str, Any]]:
        """Format retrieved documents for horizontal consumption"""
        formatted_docs = []
        
        for i, doc in enumerate(documents):
            formatted_doc = {
                "rank": i + 1,
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "page_number": doc.metadata.get("page_number", "unknown"),
                "similarity_score": getattr(doc, 'score', 0.0),
                "metadata": doc.metadata,
                "content_preview": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
            }
            formatted_docs.append(formatted_doc)
        
        return formatted_docs
    
    def _analyze_retrieval_quality(self, documents: List[Dict], query: str) -> Dict[str, Any]:
        """Analyze retrieval quality for horizontal quality assurance"""
        if not documents:
            return {"overall_score": 0, "aspects": {}}
        
        # Simple quality metrics
        content_lengths = [len(doc.get('content', '')) for doc in documents]
        avg_content_length = sum(content_lengths) / len(content_lengths)
        
        # Diversity metric (simple version)
        unique_pages = len(set(doc.get('page_number', 'unknown') for doc in documents))
        diversity_score = unique_pages / len(documents) if documents else 0
        
        return {
            "overall_score": min(10, (avg_content_length / 100 + diversity_score * 5)),
            "aspects": {
                "content_completeness": min(10, avg_content_length / 50),
                "source_diversity": min(10, diversity_score * 10),
                "result_count_adequacy": min(10, len(documents) * 2.5)
            },
            "metrics": {
                "average_content_length": round(avg_content_length, 2),
                "unique_sources": unique_pages,
                "diversity_score": round(diversity_score, 2)
            }
        }
    
    def _generate_retrieval_recommendations(self, quality_metrics: Dict) -> List[str]:
        """Generate retrieval recommendations based on quality analysis"""
        recommendations = []
        
        if quality_metrics["aspects"]["content_completeness"] < 5:
            recommendations.append("Consider increasing chunk size for more complete content")
        
        if quality_metrics["aspects"]["source_diversity"] < 5:
            recommendations.append("Try expanding search to more diverse document sections")
        
        if quality_metrics["aspects"]["result_count_adequacy"] < 5:
            recommendations.append("Increase the number of retrieved documents (k value)")
        
        return recommendations if recommendations else ["Retrieval quality is good"]
    
    def clear_vector_db(self):
        """Clear the current vector database"""
        if self.vector_db:
            self.vector_db.delete_collection()
            self.vector_db = None
            logger.info("Vector database cleared")
    
    def _calculate_processing_time(self, start_time: str) -> float:
        """Calculate processing time for performance tracking"""
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.now()
        return (end_dt - start_dt).total_seconds()
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities for horizontal system discovery"""
        return {
            "agent_name": self.name,
            "role": "Semantic search and document retrieval",
            "capabilities": [
                "semantic_search",
                "vector_database_management",
                "specialized_retrieval",
                "retrieval_quality_analysis"
            ],
            "horizontal_features": [
                "specialized_data_requests",
                "retrieval_analysis",
                "validation_services",
                "broadcast_messaging"
            ],
            "retrieval_strategies": [
                "semantic_similarity",
                "legal_enhanced_search",
                "comprehensive_search"
            ]
        }