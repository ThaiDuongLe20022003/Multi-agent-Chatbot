"""
Legal Analyzer Agent - specialized in legal analysis and interpretation
"""

import logging
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentContext, AgentResponse


logger = logging.getLogger(__name__)


class LegalAnalyzerAgent(BaseAgent):
    """Agent specialized in legal analysis, interpretation, and legal reasoning"""
    
    def __init__(self):
        super().__init__("legal_analyzer")
        self.llm = None  # Will be initialized with selected model
        
        # Legal analysis prompt template
        self.analysis_prompt = ChatPromptTemplate.from_template("""
        You are an expert legal analyst with deep knowledge of legal principles, 
        case law interpretation, and legal reasoning.

        LEGAL DOCUMENT CONTEXT:
        {context}

        USER'S LEGAL QUESTION:
        {question}

        Please provide a comprehensive legal analysis:

        1. **Legal Issues Identification**: What are the key legal issues presented?
        2. **Relevant Legal Principles**: What legal principles, statutes, or precedents apply?
        3. **Application to Facts**: How do these legal principles apply to the provided context?
        4. **Legal Reasoning**: Provide step-by-step legal reasoning.
        5. **Potential Implications**: What are the potential legal consequences or implications?

        If the context doesn't contain sufficient legal information, clearly state the limitations.

        LEGAL ANALYSIS:
        """)
    
    def initialize_model(self, model_name: str):
        """Initialize the LLM with the selected model"""
        try:
            self.llm = ChatOllama(model = model_name, temperature = 0.1, request_timeout = 120.0)
            logger.info(f"Legal analyzer agent initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize legal analyzer with model {model_name}: {e}")
            raise
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Perform legal analysis on the provided context and query"""
        try:
            if not self.llm:
                return self._create_error_response("Legal analyzer agent not initialized with a model")
                
            query = context.query
            retrieved_data = context.metadata.get("retrieved_documents", [])
            
            if not retrieved_data:
                return self._create_error_response("No retrieved documents available for legal analysis")
            
            # Prepare context from retrieved documents
            analysis_context = self._prepare_legal_context(retrieved_data)
            
            # Perform legal analysis
            legal_analysis = self._perform_legal_analysis(query, analysis_context)
            
            result_data = {
                "original_query": query,
                "legal_analysis": legal_analysis,
                "context_used": analysis_context[:1000] + "..." if len(analysis_context) > 1000 else analysis_context,
                "analysis_type": "comprehensive_legal_review"
            }
            
            logger.info("Legal analysis completed successfully")
            return self._create_success_response(result_data)
            
        except Exception as e:
            logger.error(f"Legal analysis error: {e}")
            return self._create_error_response(f"Legal analysis failed: {str(e)}")
    
    def _prepare_legal_context(self, retrieved_documents: List[Dict[str, Any]]) -> str:
        """Prepare legal context from retrieved documents"""
        context_parts = []
        
        for doc in retrieved_documents:
            content = doc.get('content', '')
            page_info = f"Page {doc.get('page_number', 'N/A')}"
            context_parts.append(f"[{page_info}] {content}")
        
        return "\n\n".join(context_parts)
    
    def _perform_legal_analysis(self, query: str, context: str) -> str:
        """Execute legal analysis using LLM"""
        try:
            prompt = self.analysis_prompt.format(
                context = context[:3000],  # Limit context length
                question = query
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM legal analysis error: {e}")
            return f"Legal analysis could not be completed due to an error: {str(e)}"