"""
PDF Processing Agent with horizontal collaboration capabilities.
Enhanced for peer-to-peer communication in multi-agent system.
"""

import logging
import tempfile
import shutil
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base_agent import BaseAgent, AgentContext, AgentResponse
from config.settings import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


logger = logging.getLogger(__name__)


class PDFProcessingAgent(BaseAgent):
    """Agent specialized in PDF text extraction and processing with horizontal collaboration"""
    
    def __init__(self):
        super().__init__("pdf_processor")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP,
            length_function=len
        )
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Process PDF file and extract text chunks with horizontal collaboration"""
        start_time = self._get_timestamp()
        collaborations = []
        
        try:
            # Check if we have PDF file in context
            pdf_file = context.metadata.get("pdf_file")
            if not pdf_file:
                return self._create_error_response("No PDF file provided in context")
            
            logger.info(f"📄 PDF Processing Agent processing: {getattr(pdf_file, 'name', 'unknown')}")
            
            # Extract text from PDF
            chunks = self._extract_pdf_text(pdf_file)
            
            # BROADCAST availability of PDF chunks to peer agents
            broadcast_responses = self.broadcast_message(
                "pdf_data_available",
                {
                    "total_chunks": len(chunks),
                    "source_file": getattr(pdf_file, 'name', 'unknown'),
                    "chunk_sample": chunks[0]['content'][:200] if chunks else ""
                }
            )
            collaborations.extend(broadcast_responses)
            
            # Prepare response data with collaboration info
            result_data = {
                "chunks": chunks,
                "total_chunks": len(chunks),
                "source_file": getattr(pdf_file, 'name', 'unknown'),
                "collaboration_requests": len(broadcast_responses)
            }
            
            processing_time = self._calculate_processing_time(start_time)
            logger.info(f"✅ PDF processing completed: {len(chunks)} chunks, {len(collaborations)} collaborations")
            
            return self._create_success_response(
                data=result_data,
                collaborations=collaborations,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ PDF processing error: {e}")
            return self._create_error_response(f"PDF processing failed: {str(e)}")
    
    def _extract_pdf_text(self, pdf_file) -> List[Dict[str, Any]]:
        """Extract and chunk PDF text with enhanced metadata for horizontal collaboration"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save uploaded file temporarily
            path = f"{temp_dir}/{pdf_file.name}"
            with open(path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Load and split PDF
            loader = PyPDFLoader(path)
            documents = loader.load_and_split()
            
            # Convert to chunks with enhanced metadata for horizontal usage
            chunks = []
            for i, doc in enumerate(documents):
                chunk_data = {
                    "content": doc.page_content,
                    "page_number": i + 1,
                    "metadata": doc.metadata,
                    "chunk_id": f"chunk_{i+1}",
                    "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "token_estimate": len(doc.page_content.split())  # For other agents to estimate processing
                }
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
            raise
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _handle_data_request(self, content: Dict) -> Dict:
        """
        Handle data requests from peer agents in horizontal collaboration
        Example: Legal Agent requests specific PDF sections
        """
        try:
            request_type = content.get("request_type", "general")
            chunk_sample = content.get("chunk_sample", [])
            
            if request_type == "legal_sections":
                # Filter chunks that might contain legal content
                legal_keywords = ['contract', 'agreement', 'clause', 'liability', 'obligation', 'rights']
                legal_chunks = [
                    chunk for chunk in chunk_sample 
                    if any(keyword in chunk.get('content', '').lower() for keyword in legal_keywords)
                ]
                
                return {
                    "status": "success",
                    "legal_chunks_found": len(legal_chunks),
                    "legal_chunks_sample": legal_chunks[:3] if legal_chunks else [],
                    "message": f"Found {len(legal_chunks)} potentially legal-relevant chunks"
                }
            
            elif request_type == "structure_analysis":
                # Provide PDF structure analysis to other agents
                total_pages = len(set(chunk.get('page_number', 0) for chunk in chunk_sample))
                avg_chunk_size = sum(len(chunk.get('content', '')) for chunk in chunk_sample) / len(chunk_sample) if chunk_sample else 0
                
                return {
                    "status": "success", 
                    "total_pages": total_pages,
                    "average_chunk_size": round(avg_chunk_size, 2),
                    "total_chunks": len(chunk_sample),
                    "analysis": "PDF structure analysis provided"
                }
            
            else:
                return {
                    "status": "success",
                    "message": "PDF processing data available",
                    "chunk_count": len(chunk_sample),
                    "capabilities": ["text_extraction", "chunking", "structure_analysis"]
                }
                
        except Exception as e:
            logger.error(f"Error handling data request: {e}")
            return {"status": "error", "message": f"Data request failed: {str(e)}"}
    
    def _handle_analysis_request(self, content: Dict) -> Dict:
        """
        Handle analysis requests from peer agents
        Example: Data Agent requests PDF content analysis
        """
        try:
            analysis_type = content.get("analysis_type", "general")
            chunks_to_analyze = content.get("chunks", [])
            
            if analysis_type == "content_density":
                # Analyze content density for retrieval optimization
                if not chunks_to_analyze:
                    return {"status": "error", "message": "No chunks provided for analysis"}
                
                content_lengths = [len(chunk.get('content', '')) for chunk in chunks_to_analyze]
                word_counts = [len(chunk.get('content', '').split()) for chunk in chunks_to_analyze]
                
                return {
                    "status": "success",
                    "analysis_type": "content_density",
                    "total_chunks_analyzed": len(chunks_to_analyze),
                    "average_content_length": round(sum(content_lengths) / len(content_lengths), 2),
                    "average_word_count": round(sum(word_counts) / len(word_counts), 2),
                    "max_content_length": max(content_lengths),
                    "min_content_length": min(content_lengths)
                }
            
            else:
                return {
                    "status": "success",
                    "message": "PDF analysis capabilities available",
                    "available_analyses": ["content_density", "structure_analysis", "keyword_analysis"]
                }
                
        except Exception as e:
            logger.error(f"Error handling analysis request: {e}")
            return {"status": "error", "message": f"Analysis request failed: {str(e)}"}
    
    def _handle_collaboration_request(self, content: Dict) -> Dict:
        """
        Handle collaboration requests from peer agents
        Example: Quality Assurance Agent requests PDF quality metrics
        """
        try:
            collaboration_type = content.get("collaboration_type", "info")
            
            if collaboration_type == "quality_metrics":
                # Provide PDF processing quality metrics
                return {
                    "status": "success",
                    "agent_name": self.name,
                    "capabilities": [
                        "pdf_text_extraction",
                        "document_chunking", 
                        "metadata_enrichment",
                        "structure_analysis"
                    ],
                    "performance_metrics": {
                        "average_processing_time": "2-5 seconds",
                        "max_file_size": "50MB",
                        "supported_formats": ["PDF"],
                        "chunking_strategy": "RecursiveCharacterTextSplitter"
                    }
                }
            
            else:
                return {
                    "status": "success",
                    "message": "PDF Processing Agent collaboration available",
                    "agent_role": "Extract and prepare PDF content for multi-agent processing"
                }
                
        except Exception as e:
            logger.error(f"Error handling collaboration request: {e}")
            return {"status": "error", "message": f"Collaboration request failed: {str(e)}"}
    
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
            "role": "PDF text extraction and document preparation",
            "capabilities": [
                "pdf_text_extraction",
                "document_chunking", 
                "metadata_enrichment",
                "structure_analysis",
                "peer_collaboration"
            ],
            "horizontal_features": [
                "data_request_handling",
                "analysis_request_handling", 
                "collaboration_request_handling",
                "broadcast_messaging"
            ]
        }