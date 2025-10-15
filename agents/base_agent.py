"""
Base agent interface that all specialized agents will implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from dataclasses import dataclass


@dataclass
class AgentContext:
    """Shared context passed between agents"""
    query: str
    session_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass  
class AgentResponse:
    """Standardized response from any agent"""
    success: bool
    data: Any
    error_message: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True
    
    @abstractmethod
    def process(self, context: AgentContext) -> AgentResponse:
        """Main processing method that all agents must implement"""
        pass
    
    def health_check(self) -> bool:
        """Check if agent is functioning properly"""
        return self.is_healthy
    
    def _create_success_response(self, data: Any, metadata: Dict = None) -> AgentResponse:
        """Helper method for successful responses"""
        return AgentResponse(
            success = True,
            data = data,
            metadata = metadata or {}
        )
    
    def _create_error_response(self, error_message: str) -> AgentResponse:
        """Helper method for error responses"""
        return AgentResponse(
            success = False,
            data = None,
            error_message = error_message
        )