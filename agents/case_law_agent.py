"""
Case Law Specialist Agent - Precedents and judicial reasoning.
"""

from typing import List
from agents.base_agent import BaseLegalAgent

class CaseLawAgent(BaseLegalAgent):
    """Specializes in case law, precedents, and judicial reasoning"""
    
    def __init__(self, model: str = None):
        super().__init__("case_law", model)
    
    def _get_analysis_prompt(self, question: str, context: str) -> str:
        return f"""
        As a Case Law Specialist, analyze the following legal question:
        
        QUESTION: {question}
        
        CONTEXT FROM DOCUMENTS:
        {context}
        
        Provide case law analysis focusing on:
        - Relevant legal precedents and citations
        - Judicial reasoning patterns
        - Fact pattern similarities and distinctions
        - Potential outcomes based on case law
        - Strengths and weaknesses of legal arguments
        
        Emphasize precedent-based reasoning and judicial interpretation.
        
        ANALYSIS:
        """
    
    def _get_domain_keywords(self) -> List[str]:
        return [
            "precedent", "case law", "judicial", "court", "ruling",
            "opinion", "appeal", "jurisdiction", "citation", "precedential",
            "distinguish", "analogous", "holding", "dicta", "stare decisis",
            "judgment", "verdict", "legal decision", "court case", "precedent-setting"
        ]