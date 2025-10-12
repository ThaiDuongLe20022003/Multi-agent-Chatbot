"""
Agents package for multi-agent legal analysis.
"""

from .base_agent import BaseLegalAgent
from .legal_research_agent import LegalResearchAgent
from .case_law_agent import CaseLawAgent

__all__ = [
    'BaseLegalAgent',
    'LegalResearchAgent', 
    'CaseLawAgent'
]