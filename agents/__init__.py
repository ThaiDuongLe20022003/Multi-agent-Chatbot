"""
Multi-agent system for horizontal collaboration.
"""

from .base_agent import BaseAgent
from .agent_manager import AgentManager
from .pdf_processing_agent import PDFProcessingAgent
from .data_retrieval_agent import DataRetrievalAgent

__all__ = [
    'BaseAgent',
    'AgentManager', 
    'PDFProcessingAgent',
    'DataRetrievalAgent'
]