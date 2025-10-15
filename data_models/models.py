"""
Data models for the DeepLaw RAG application.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class LLMEvaluation:
    """Comprehensive LLM evaluation metrics using LLM-as-a-judge"""
    faithfulness: float  # 0-10: Does the answer rely on the provided context?
    groundedness: float  # 0-10: Can information be traced back to context?
    factual_consistency: float  # 0-10: Factual alignment with context
    relevance: float  # 0-10: Addresses the actual query
    completeness: float  # 0-10: Covers all important aspects
    fluency: float  # 0-10: Natural, coherent, and well-written
    overall_score: float  # 0-10: Overall quality score
    evaluation_notes: str  # Detailed explanation from judge
    judge_model: str  # Which model performed this evaluation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "faithfulness": self.faithfulness,
            "groundedness": self.groundedness,
            "factual_consistency": self.factual_consistency,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "fluency": self.fluency,
            "overall_score": self.overall_score,
            "evaluation_notes": self.evaluation_notes,
            "judge_model": self.judge_model
        }


@dataclass
class LLMMetrics:
    """Data class to store LLM performance metrics"""
    timestamp: str
    query: str
    response: str
    context: str
    response_time: float
    token_count: int
    tokens_per_second: float
    model: str
    session_id: str
    evaluations: List[LLMEvaluation]  # Multiple evaluations from different judges


@dataclass 
class ChatMessage:
    """Data class for chat messages"""
    role: str  # 'user' or 'assistant'
    content: str
    evaluations: List[Dict[str, Any]] = field(default_factory = list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for compatibility"""
        return {
            "role": self.role,
            "content": self.content,
            "evaluations": self.evaluations
        }