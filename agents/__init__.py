"""
Multi-agent system for horizontal collaboration.
"""

from .base_agent import BaseAgent, AgentContext, AgentResponse
from .agent_manager import AgentManager, WorkflowResult
from .pdf_processing_agent import PDFProcessingAgent
from .data_retrieval_agent import DataRetrievalAgent
from .legal_analyzer_agent import LegalAnalyzerAgent
from .summarize_reason_agent import SummarizeReasonAgent

__all__ = [
    'BaseAgent',
    'AgentContext', 
    'AgentResponse',
    'AgentManager',
    'WorkflowResult',
    'PDFProcessingAgent',
    'DataRetrievalAgent',
    'LegalAnalyzerAgent', 
    'SummarizeReasonAgent'
]