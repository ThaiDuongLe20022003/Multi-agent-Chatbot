"""
Summarize & Reason Agent - synthesizes information and provides logical reasoning
"""

import logging
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentContext, AgentResponse


logger = logging.getLogger(__name__)


class SummarizeReasonAgent(BaseAgent):
    """Agent specialized in synthesis, summarization, and logical reasoning"""
    
    def __init__(self):
        super().__init__("summarize_reason")
        self.llm = None  # Will be initialized with selected model
        
        # Synthesis prompt template
        self.synthesis_prompt = ChatPromptTemplate.from_template("""
        You are an expert at synthesizing information and providing clear, logical reasoning.

        ORIGINAL QUERY:
        {query}

        LEGAL ANALYSIS:
        {legal_analysis}

        SUPPORTING DOCUMENTS:
        {supporting_docs}

        Your task is to synthesize this information into a comprehensive, well-reasoned response:

        1. **Executive Summary**: Begin with a concise summary of the key findings.
        2. **Logical Flow**: Present information in a logical, easy-to-follow structure.
        3. **Key Points**: Highlight the most important legal points and insights.
        4. **Practical Implications**: Explain what this means in practical terms.
        5. **Confidence Level**: Indicate the confidence level based on available information.

        Ensure the response is professional, clear, and directly addresses the original query.

        SYNTHESIZED RESPONSE:
        """)
    
    def initialize_model(self, model_name: str):
        """Initialize the LLM with the selected model"""
        try:
            self.llm = ChatOllama(model = model_name, temperature = 0.1, request_timeout = 120.0)
            logger.info(f"Summarize reason agent initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize summarize reason agent with model {model_name}: {e}")
            raise
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Synthesize and reason across multiple information sources"""
        try:
            if not self.llm:
                return self._create_error_response("Summarize reason agent not initialized with a model")
                
            query = context.query
            legal_analysis = context.metadata.get("legal_analysis")
            retrieved_docs = context.metadata.get("retrieved_documents", [])
            
            if not legal_analysis:
                return self._create_error_response("No legal analysis available for synthesis")
            
            # Prepare supporting documents context
            supporting_context = self._prepare_supporting_context(retrieved_docs)
            
            # Synthesize comprehensive response
            synthesized_response = self._synthesize_response(
                query, legal_analysis, supporting_context
            )
            
            result_data = {
                "original_query": query,
                "synthesized_response": synthesized_response,
                "sources_used": {
                    "legal_analysis": True,
                    "supporting_documents": len(retrieved_docs)
                },
                "response_type": "comprehensive_synthesis"
            }
            
            logger.info("Synthesis and reasoning completed successfully")
            return self._create_success_response(result_data)
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self._create_error_response(f"Synthesis failed: {str(e)}")
    
    def _prepare_supporting_context(self, retrieved_documents: List[Dict[str, Any]]) -> str:
        """Prepare supporting context from retrieved documents"""
        if not retrieved_documents:
            return "No supporting documents available."
        
        context_parts = ["SUPPORTING DOCUMENTS:"]
        for i, doc in enumerate(retrieved_documents[:3]):  # Use top 3 most relevant
            content_preview = doc.get('content', '')[:300]
            page_info = f"Page {doc.get('page_number', 'N/A')}"
            context_parts.append(f"{i+1}. [{page_info}] {content_preview}...")
        
        return "\n".join(context_parts)
    
    def _synthesize_response(self, query: str, legal_analysis: str, supporting_docs: str) -> str:
        """Synthesize comprehensive response using LLM"""
        try:
            # If legal_analysis is a dictionary (from LegalAnalyzerAgent), extract the text
            if isinstance(legal_analysis, dict):
                legal_text = legal_analysis.get('legal_analysis', str(legal_analysis))
            else:
                legal_text = legal_analysis
            
            prompt = self.synthesis_prompt.format(
                query = query,
                legal_analysis = legal_text[:2000],  # Limit length
                supporting_docs = supporting_docs
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")
            return f"Synthesis could not be completed due to an error: {str(e)}"