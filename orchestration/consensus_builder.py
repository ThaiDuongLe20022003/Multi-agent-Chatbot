"""
Builds consensus from multiple agent analyses.
"""

from typing import List, Dict, Any
import logging

from data_models.agent_models import AgentAnalysis, MultiAgentResponse

logger = logging.getLogger(__name__)

class ConsensusBuilder:
    """Synthesizes final response from multiple agent analyses"""
    
    def __init__(self, model: str = None):
        self.model = model
    
    def build_consensus(self, agent_analyses: List[AgentAnalysis], 
                       question: str, context: str) -> MultiAgentResponse:
        """Build consensus response from multiple agent analyses"""
        logger.info(f"Building consensus from {len(agent_analyses)} agent analyses")
        
        # Prepare analysis summary for consensus building
        analysis_summary = self._prepare_analysis_summary(agent_analyses)
        
        # Generate consensus response
        consensus_prompt = self._get_consensus_prompt(question, context, analysis_summary)
        
        # Use simple LLM call for consensus
        try:
            from processing.rag_chain import simple_llm_call
            final_response = simple_llm_call(consensus_prompt, self.model)
        except ImportError:
            final_response = self._simple_consensus(agent_analyses, question)
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(agent_analyses)
        
        return MultiAgentResponse(
            final_response = final_response,
            agent_analyses = agent_analyses,
            communication_log = [],  # Will be populated in Phase 2
            consensus_score = consensus_score
        )
    
    def _prepare_analysis_summary(self, agent_analyses: List[AgentAnalysis]) -> str:
        """Prepare summary of all agent analyses"""
        summary_parts = []
        
        for analysis in agent_analyses:
            summary_parts.append(f"""
            {analysis.agent_type.upper()} ANALYSIS (Confidence: {analysis.confidence:.1%}):
            {analysis.analysis}
            
            Supporting Evidence:
            {chr(10).join(['• ' + evidence for evidence in analysis.supporting_evidence[:2]])}
            """)
        
        return "\n".join(summary_parts)
    
    def _get_consensus_prompt(self, question: str, context: str, analysis_summary: str) -> str:
        return f"""
        Synthesize a comprehensive legal response by combining insights from multiple legal specialists:
        
        ORIGINAL QUESTION: {question}
        
        CONTEXT FROM DOCUMENTS:
        {context}
        
        SPECIALIST ANALYSES:
        {analysis_summary}
        
        Please create a unified, coherent response that:
        1. Integrates the key insights from all specialists
        2. Resolves any contradictions or differing perspectives
        3. Provides a balanced, comprehensive answer to the original question
        4. Highlights areas of strong consensus and notes any remaining uncertainties
        
        FINAL SYNTHESIZED RESPONSE:
        """
    
    def _simple_consensus(self, agent_analyses: List[AgentAnalysis], question: str) -> str:
        """Simple consensus fallback"""
        responses = [f"## {analysis.agent_type.replace('_', ' ').title()} Perspective\n{analysis.analysis}" 
                    for analysis in agent_analyses]
        
        return f"""
        # Multi-Agent Analysis: {question}
        
        {' '.join(responses)}
        
        ## Consensus Summary
        The specialist agents have provided complementary perspectives on this legal question.
        """
    
    def _calculate_consensus_score(self, agent_analyses: List[AgentAnalysis]) -> float:
        """Calculate how much the agents agree (simplified for Phase 1)"""
        if len(agent_analyses) < 2:
            return 1.0
        
        # Simple consensus calculation based on confidence
        avg_confidence = sum(analysis.confidence for analysis in agent_analyses) / len(agent_analyses)
        return min(avg_confidence * 0.7 + 0.3, 1.0)