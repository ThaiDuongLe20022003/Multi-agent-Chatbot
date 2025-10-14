"""
Multi-agent RAG chain building on existing system.
"""

import logging
from typing import Tuple, Dict, Any

from agents.legal_research_agent import LegalResearchAgent
from agents.case_law_agent import CaseLawAgent
from orchestration.communication_hub import AgentCommunicationHub
from orchestration.consensus_builder import ConsensusBuilder
from data_models.agent_models import MultiAgentResponse

logger = logging.getLogger(__name__)


def process_question_multi_agent(question: str, vector_db, selected_model: str) -> Tuple[str, str, Dict[str, Any]]:
    """Process question using horizontal multi-agent system"""
    logger.info(f"Multi-agent processing: {question}")
    
    # 1. Retrieve context (using your existing system)
    from processing.rag_chain import process_question_simple
    context, response = process_question_simple(question, vector_db, selected_model)  # ← Updated assignment
    
    # 2. Set up agent team and communication
    hub = AgentCommunicationHub()
    
    legal_agent = LegalResearchAgent(model = selected_model)
    case_agent = CaseLawAgent(model = selected_model)
    
    hub.register_agent(legal_agent)
    hub.register_agent(case_agent)
    
    # 3. Have agents analyze in parallel (with simulated communication)
    legal_analysis = legal_agent.analyze(question, context)
    case_analysis = case_agent.analyze(question, context)
    
    # 4. Build consensus
    consensus_builder = ConsensusBuilder(model = selected_model)
    multi_agent_response = consensus_builder.build_consensus(
        [legal_analysis, case_analysis], question, context
    )
    
    # 5. Prepare return values compatible with your existing system
    final_response = multi_agent_response.final_response
    
    # Prepare agent analyses for display
    agent_analyses_formatted = []
    for analysis in multi_agent_response.agent_analyses:
        agent_analyses_formatted.append({
            "agent_type": analysis.agent_type,
            "analysis": analysis.analysis,
            "confidence": analysis.confidence,
            "refined_analysis": analysis.refined_analysis
        })

    return context, final_response, {  
        "multi_agent": True,
        "agent_analyses": agent_analyses_formatted,
        "consensus_score": multi_agent_response.consensus_score,
        "communication_log": hub.message_bus
    }