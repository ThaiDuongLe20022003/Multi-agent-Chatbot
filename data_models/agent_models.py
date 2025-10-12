"""
Data models for multi-agent communication and coordination.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

class MessageType(Enum):
    INITIAL_ANALYSIS = "initial_analysis"
    PEER_REVIEW = "peer_review" 
    CLARIFICATION_REQUEST = "clarification_request"
    CONSENSUS_BUILDING = "consensus_building"
    FINAL_SYNTHESIS = "final_synthesis"

@dataclass
class AgentMessage:
    """Message between agents in the horizontal network"""
    sender: str
    receiver: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str
    conversation_id: str

@dataclass
class AgentAnalysis:
    """Analysis result from a single agent"""
    agent_type: str
    analysis: str
    confidence: float
    supporting_evidence: List[str]
    questions_for_peers: List[str] = field(default_factory = list)
    peer_feedback: List[Dict] = field(default_factory = list)
    refined_analysis: Optional[str] = None

@dataclass
class MultiAgentResponse:
    """Complete response from multi-agent system"""
    final_response: str
    agent_analyses: List[AgentAnalysis]
    communication_log: List[AgentMessage]
    consensus_score: float