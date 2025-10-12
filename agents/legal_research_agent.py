"""
Legal Research Specialist Agent - General legal document analysis.
"""

from typing import List
from agents.base_agent import BaseLegalAgent

class LegalResearchAgent(BaseLegalAgent):
    """Specializes in general legal document analysis"""
    
    def __init__(self, model: str = None):
        super().__init__("legal_research", model)
    
    def _get_analysis_prompt(self, question: str, context: str) -> str:
        return f"""
        As a Legal Research Specialist, analyze the following legal question:
        
        QUESTION: {question}
        
        CONTEXT FROM DOCUMENTS:
        {context}
        
        Provide comprehensive legal analysis covering:
        - Key legal principles and concepts
        - Document structure and organization
        - Main arguments and reasoning
        - General legal implications
        - Potential areas needing further research
        
        Focus on thorough, well-reasoned legal analysis.
        
        ANALYSIS:
        """
    
    def _get_domain_keywords(self) -> List[str]:
        return [
            "legal", "principle", "doctrine", "jurisdiction", "precedent",
            "analysis", "interpretation", "application", "requirement",
            "obligation", "right", "liability", "enforcement", "statute",
            "regulation", "compliance", "legal framework", "law", "legal doctrine"
        ]