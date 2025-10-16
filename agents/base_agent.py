"""
Base agent interface with horizontal communication capabilities.
Enhanced for true peer-to-peer collaboration in multi-agent systems.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass
import uuid
import time


@dataclass
class AgentMessage:
    """Message for inter-agent communication in horizontal architecture"""
    message_id: str
    sender: str
    receiver: str
    message_type: str  # 'data_request', 'analysis_result', 'clarification', 'collaboration'
    content: Dict[str, Any]
    timestamp: str
    priority: int = 1
    requires_response: bool = True
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())


@dataclass
class AgentContext:
    """Shared context with horizontal collaboration capabilities"""
    query: str
    session_id: str
    metadata: Dict[str, Any] = None
    collaborative_data: Dict[str, Any] = None  # Data from peer agents
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.collaborative_data is None:
            self.collaborative_data = {}


@dataclass  
class AgentResponse:
    """Standardized response with collaboration tracking"""
    success: bool
    data: Any
    error_message: str = None
    metadata: Dict[str, Any] = None
    collaborations: List[AgentMessage] = None  # Track inter-agent communications
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.collaborations is None:
            self.collaborations = []


class BaseAgent(ABC):
    """Abstract base class with horizontal communication capabilities"""
    
    def __init__(self, name: str):
        self.name = name
        self.is_healthy = True
        self.peer_agents: Dict[str, 'BaseAgent'] = {}  # Direct peer references for horizontal comm
        self.message_handlers: Dict[str, Callable] = {}
        self.collaboration_history: List[AgentMessage] = []
        self._setup_message_handlers()
    
    def register_peer(self, agent_name: str, agent: 'BaseAgent'):
        """Register a peer agent for direct horizontal communication"""
        self.peer_agents[agent_name] = agent
        print(f"🤝 {self.name} connected to {agent_name}")
    
    def _setup_message_handlers(self):
        """Setup default message handlers for horizontal communication"""
        self.message_handlers = {
            'data_request': self._handle_data_request,
            'analysis_request': self._handle_analysis_request,
            'clarification': self._handle_clarification,
            'validation_request': self._handle_validation_request,
            'collaboration_request': self._handle_collaboration_request
        }
    
    def send_message(self, receiver: str, message_type: str, content: Dict[str, Any], requires_response: bool = True) -> Optional[AgentMessage]:
        """Send message to peer agent directly - HORIZONTAL COMMUNICATION"""
        if receiver in self.peer_agents:
            message = AgentMessage(
                message_id=str(uuid.uuid4()),
                sender=self.name,
                receiver=receiver,
                message_type=message_type,
                content=content,
                timestamp=self._get_timestamp(),
                requires_response=requires_response
            )
            
            # Direct method call to peer agent - TRUE HORIZONTAL
            response = self.peer_agents[receiver].receive_message(message)
            self.collaboration_history.append(message)
            if response:
                self.collaboration_history.append(response)
            return response
        else:
            print(f"⚠️ {self.name} cannot find peer agent: {receiver}")
            return None
    
    def broadcast_message(self, message_type: str, content: Dict[str, Any]) -> List[AgentMessage]:
        """Broadcast message to all peer agents - GROUP COLLABORATION"""
        responses = []
        for agent_name in self.peer_agents:
            if agent_name != self.name:  # Don't send to self
                response = self.send_message(agent_name, message_type, content, requires_response=False)
                if response:
                    responses.append(response)
        return responses
    
    def receive_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Receive and process message from peer agent"""
        try:
            handler = self.message_handlers.get(message.message_type)
            if handler:
                result = handler(message.content)
                if message.requires_response:
                    response = AgentMessage(
                        message_id=str(uuid.uuid4()),
                        sender=self.name,
                        receiver=message.sender,
                        message_type=f"{message.message_type}_response",
                        content=result,
                        timestamp=self._get_timestamp()
                    )
                    return response
                return None
            else:
                return self._create_error_response(message, f"No handler for message type: {message.message_type}")
                
        except Exception as e:
            return self._create_error_response(message, f"Error processing message: {str(e)}")
    
    # Default message handlers for horizontal communication
    def _handle_data_request(self, content: Dict) -> Dict:
        return {"status": "not_implemented", "message": "Data request handler not implemented"}
    
    def _handle_analysis_request(self, content: Dict) -> Dict:
        return {"status": "not_implemented", "message": "Analysis request handler not implemented"}
    
    def _handle_clarification(self, content: Dict) -> Dict:
        return {"status": "not_implemented", "message": "Clarification handler not implemented"}
    
    def _handle_validation_request(self, content: Dict) -> Dict:
        return {"status": "not_implemented", "message": "Validation request handler not implemented"}
    
    def _handle_collaboration_request(self, content: Dict) -> Dict:
        return {"status": "not_implemented", "message": "Collaboration request handler not implemented"}
    
    def _create_error_response(self, original_message: AgentMessage, error: str) -> AgentMessage:
        return AgentMessage(
            message_id=str(uuid.uuid4()),
            sender=self.name,
            receiver=original_message.sender,
            message_type="error",
            content={"error": error, "original_message": original_message.content},
            timestamp=self._get_timestamp()
        )
    
    @abstractmethod
    def process(self, context: AgentContext) -> AgentResponse:
        """Main processing method that all agents must implement"""
        pass
    
    def process_async(self, context: AgentContext) -> 'AgentResponse':
        """Async processing for parallel execution in horizontal workflow"""
        return self.process(context)
    
    def health_check(self) -> bool:
        """Check if agent is functioning properly"""
        return self.is_healthy
    
    def _create_success_response(self, data: Any, metadata: Dict = None, collaborations: List = None, processing_time: float = 0.0) -> AgentResponse:
        """Helper method for successful responses with collaboration tracking"""
        return AgentResponse(
            success=True,
            data=data,
            metadata=metadata or {},
            collaborations=collaborations or [],
            processing_time=processing_time
        )
    
    def _create_error_response(self, error_message: str) -> AgentResponse:
        """Helper method for error responses"""
        return AgentResponse(
            success=False,
            data=None,
            error_message=error_message
        )
    
    def _get_timestamp(self):
        """Get current timestamp for collaboration tracking"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get statistics about agent collaborations for monitoring"""
        sent_messages = [msg for msg in self.collaboration_history if msg.sender == self.name]
        received_messages = [msg for msg in self.collaboration_history if msg.receiver == self.name]
        
        return {
            "total_collaborations": len(self.collaboration_history),
            "messages_sent": len(sent_messages),
            "messages_received": len(received_messages),
            "peer_agents": list(self.peer_agents.keys())
        }