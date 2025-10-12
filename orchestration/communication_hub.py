"""
Central communication hub for horizontal agent coordination.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime

from data_models.agent_models import AgentMessage, MessageType

logger = logging.getLogger(__name__)

class AgentCommunicationHub:
    """Facilitates communication between agents in horizontal architecture"""
    
    def __init__(self):
        self.agents = {}
        self.message_bus = []
        self.conversation_counter = 0
    
    def register_agent(self, agent):
        """Register an agent with the communication hub"""
        self.agents[agent.expertise] = agent
        agent.set_communication_hub(self)
        logger.info(f"Registered agent: {agent.expertise}")
    
    def broadcast_message(self, sender: str, message_type: MessageType, content: Dict[str, Any]):
        """Broadcast message to all other agents"""
        conversation_id = f"conv_{self.conversation_counter}"
        
        message = AgentMessage(
            sender = sender,
            receiver = "all",
            message_type = message_type,
            content = content,
            timestamp = datetime.now().isoformat(),
            conversation_id = conversation_id
        )
        
        self.message_bus.append(message)
        logger.info(f"Broadcast from {sender}: {message_type.value}")
        
        # In Phase 1, we'll log but not route real-time
        # In Phase 2, this will trigger real agent responses
    
    def direct_message(self, sender: str, receiver: str, message: Dict[str, Any]):
        """Send direct message to specific agent"""
        if receiver in self.agents:
            # In Phase 1, we'll just log this
            logger.info(f"Direct message from {sender} to {receiver}")
            # In Phase 2, this will trigger agent.receive_message()
    
    def get_conversation_log(self, conversation_id: str) -> List[AgentMessage]:
        """Get all messages for a conversation"""
        return [msg for msg in self.message_bus if msg.conversation_id == conversation_id]
    
    def get_agent_messages(self, agent_name: str) -> List[AgentMessage]:
        """Get all messages for a specific agent"""
        return [msg for msg in self.message_bus if msg.receiver == "all" or msg.receiver == agent_name]