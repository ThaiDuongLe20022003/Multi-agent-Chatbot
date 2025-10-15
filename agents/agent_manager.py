"""
Centralized agent manager for coordinating the multi-agent system.
"""

import logging
from typing import Dict, List, Any
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentContext, AgentResponse


logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of a complete workflow execution"""
    success: bool
    final_response: Any
    agent_responses: Dict[str, AgentResponse]
    errors: List[str]


class AgentManager:
    """Orchestrates agent workflow and handles coordination"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_history: List[WorkflowResult] = []
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the manager"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def get_agent(self, agent_name: str) -> BaseAgent:
        """Get registered agent by name"""
        return self.agents.get(agent_name)
    
    def health_check_all(self) -> Dict[str, bool]:
        """Check health of all registered agents"""
        return {name: agent.health_check() for name, agent in self.agents.items()}
    
    def execute_workflow(self, query: str, session_id: str = "default") -> WorkflowResult:
        """
        Execute the basic PDF → Data Retrieval workflow
        This will evolve as we add more agents
        """
        logger.info(f"Executing workflow for query: {query}")
        
        context = AgentContext(query=query, session_id=session_id)
        agent_responses = {}
        errors = []
        
        try:
            # Step 1: PDF Processing (if we have PDF context)
            if "pdf_processor" in self.agents and context.metadata.get("has_pdf"):
                pdf_agent = self.agents["pdf_processor"]
                pdf_result = pdf_agent.process(context)
                agent_responses["pdf_processing"] = pdf_result
                
                if not pdf_result.success:
                    errors.append(f"PDF Processing failed: {pdf_result.error_message}")
                    return WorkflowResult(
                        success = False,
                        final_response = "Error processing PDF",
                        agent_responses = agent_responses,
                        errors = errors
                    )
                
                # Add PDF data to context for next agents
                context.metadata["pdf_data"] = pdf_result.data
            
            # Step 2: Data Retrieval
            if "data_retriever" in self.agents:
                data_agent = self.agents["data_retriever"]
                data_result = data_agent.process(context)
                agent_responses["data_retrieval"] = data_result
                
                if not data_result.success:
                    errors.append(f"Data Retrieval failed: {data_result.error_message}")
                else:
                    # For now, use data retrieval result as final response
                    final_response = data_result.data
                    
                    return WorkflowResult(
                        success = True,
                        final_response = final_response,
                        agent_responses = agent_responses,
                        errors = errors
                    )
            else:
                errors.append("Data Retrieval agent not available")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            errors.append(f"System error: {str(e)}")
        
        return WorkflowResult(
            success = False,
            final_response = "Workflow execution failed",
            agent_responses = agent_responses,
            errors = errors
        )
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about workflow executions"""
        if not self.workflow_history:
            return {"total_executions": 0, "success_rate": 0.0}
        
        total = len(self.workflow_history)
        successful = sum(1 for result in self.workflow_history if result.success)
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "last_execution": self.workflow_history[-1] if self.workflow_history else None
        }