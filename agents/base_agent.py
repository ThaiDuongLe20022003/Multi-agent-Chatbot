"""
Base class for all legal specialist agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import logging
from datetime import datetime

from data_models.agent_models import AgentAnalysis

logger = logging.getLogger(__name__)

class BaseLegalAgent(ABC):
    """Base class for all legal specialist agents"""
    
    def __init__(self, expertise: str, model: str = None):
        self.expertise = expertise
        self.model = model
        self.communication_hub = None
        self.analysis_history = []
    
    def set_communication_hub(self, hub):
        """Set the communication hub for agent coordination"""
        self.communication_hub = hub
    
    def analyze(self, question: str, context: str) -> AgentAnalysis:
        """Perform analysis and communicate with peers"""
        logger.info(f"Agent {self.expertise} analyzing question: {question[:100]}...")
        
        # 1. Perform initial analysis
        initial_analysis = self._perform_initial_analysis(question, context)
        
        # 2. Share analysis with peers
        if self.communication_hub:
            self._broadcast_analysis(question, initial_analysis)
        
        # 3. Collect peer responses (simplified for Phase 1)
        peer_feedback = self._collect_peer_feedback()
        
        # 4. Refine analysis based on peer feedback
        refined_analysis = self._refine_analysis(initial_analysis, peer_feedback, question, context)
        
        return AgentAnalysis(
            agent_type = self.expertise,
            analysis = refined_analysis,
            confidence = self._calculate_confidence(refined_analysis),
            supporting_evidence = self._extract_evidence(context),
            peer_feedback = peer_feedback,
            refined_analysis = refined_analysis
        )
    
    def _perform_initial_analysis(self, question: str, context: str) -> str:
        """Perform domain-specific analysis"""
        prompt = self._get_analysis_prompt(question, context)
        
        # Use a simple LLM call
        try:
            from processing.rag_chain import simple_llm_call
            return simple_llm_call(prompt, self.model)
        except ImportError:
            # Fallback for testing
            return f"{self.expertise} analysis for: {question}\n\nBased on context: {context[:200]}..."
    
    @abstractmethod
    def _get_analysis_prompt(self, question: str, context: str) -> str:
        """Get domain-specific prompt template"""
        pass
    
    def _broadcast_analysis(self, question: str, analysis: str):
        """Broadcast analysis to other agents"""
        if self.communication_hub:
            from data_models.agent_models import MessageType
            self.communication_hub.broadcast_message(
                self.expertise,
                MessageType.INITIAL_ANALYSIS,
                {
                    "question": question,
                    "analysis": analysis,
                    "timestamp": self._get_timestamp()
                }
            )
    
    def _collect_peer_feedback(self) -> List[Dict]:
        """Collect feedback from peers (simplified for Phase 1)"""
        # In Phase 1, we'll simulate peer feedback
        return [
            {
                "from_agent": "simulated_peer",
                "feedback": "This analysis appears comprehensive from my perspective.",
                "suggestions": ["Consider adding more specific legal references."]
            }
        ]
    
    def _refine_analysis(self, initial_analysis: str, peer_feedback: List[Dict], 
                        question: str, context: str) -> str:
        """Refine analysis based on peer feedback"""
        if not peer_feedback:
            return initial_analysis
        
        feedback_text = "\n".join([
            f"- {fb['from_agent']}: {fb['feedback']} Suggestions: {', '.join(fb.get('suggestions', []))}"
            for fb in peer_feedback
        ])
        
        refinement_prompt = f"""
        Refine your legal analysis based on peer feedback:
        
        ORIGINAL QUESTION: {question}
        
        YOUR INITIAL ANALYSIS:
        {initial_analysis}
        
        PEER FEEDBACK:
        {feedback_text}
        
        Please provide an improved analysis that incorporates the valuable feedback.
        Maintain your expertise as a {self.expertise} specialist.
        
        REFINED ANALYSIS:
        """
        
        try:
            from processing.rag_chain import simple_llm_call
            return simple_llm_call(refinement_prompt, self.model)
        except ImportError:
            return f"REFINED: {initial_analysis}\n\nIncorporated feedback: {feedback_text}"
    
    def _calculate_confidence(self, analysis: str) -> float:
        """Calculate confidence score for analysis (simplified)"""
        # Simple heuristic - can be enhanced later
        length_factor = min(len(analysis) / 500, 1.0)
        keyword_matches = self._count_keyword_matches(analysis)
        return min(0.3 + (length_factor * 0.4) + (keyword_matches * 0.3), 1.0)
    
    def _count_keyword_matches(self, analysis: str) -> float:
        """Count relevant keyword matches (domain-specific)"""
        keywords = self._get_domain_keywords()
        if not keywords:
            return 0.5
        matches = sum(1 for keyword in keywords if keyword.lower() in analysis.lower())
        return matches / len(keywords)
    
    @abstractmethod
    def _get_domain_keywords(self) -> List[str]:
        """Get domain-specific keywords for confidence calculation"""
        pass
    
    def _extract_evidence(self, context: str) -> List[str]:
        """Extract supporting evidence from context"""
        # Simple evidence extraction
        sentences = context.split('.')
        return [s.strip() + '.' for s in sentences[:3] if len(s.strip()) > 10]
    
    def _get_timestamp(self):
        return datetime.now().isoformat()