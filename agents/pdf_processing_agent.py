"""
PDF Processing Agent - handles PDF text extraction and chunking
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
    """Agent specialized in PDF text extraction and processing"""
    
    def __init__(self):
        super().__init__("pdf_processor")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = DEFAULT_CHUNK_SIZE,
            chunk_overlap = DEFAULT_CHUNK_OVERLAP,
            length_function = len
        )
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Process PDF file and extract text chunks"""
        try:
            # Check if we have PDF file in context
            pdf_file = context.metadata.get("pdf_file")
            if not pdf_file:
                return self._create_error_response("No PDF file provided in context")
            
            logger.info(f"Processing PDF: {getattr(pdf_file, 'name', 'unknown')}")
            
            # Extract text from PDF
            chunks = self._extract_pdf_text(pdf_file)
            
            # Prepare response data
            result_data = {
                "chunks": chunks,
                "total_chunks": len(chunks),
                "source_file": getattr(pdf_file, 'name', 'unknown')
            }
            
            logger.info(f"PDF processing completed: {len(chunks)} chunks extracted")
            return self._create_success_response(result_data)
            
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
            return self._create_error_response(f"PDF processing failed: {str(e)}")
    
    def _extract_pdf_text(self, pdf_file) -> List[Dict[str, Any]]:
        """Extract and chunk PDF text"""
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save uploaded file temporarily
            path = f"{temp_dir}/{pdf_file.name}"
            with open(path, "wb") as f:
                f.write(pdf_file.getvalue())
            
            # Load and split PDF
            loader = PyPDFLoader(path)
            documents = loader.load_and_split()
            
            # Convert to chunks with metadata
            chunks = []
            for i, doc in enumerate(documents):
                chunk_data = {
                    "content": doc.page_content,
                    "page_number": i + 1,
                    "metadata": doc.metadata,
                    "chunk_id": f"chunk_{i+1}"
                }
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"PDF text extraction error: {e}")
            raise
        finally:
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors = True)