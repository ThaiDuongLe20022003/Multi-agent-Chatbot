"""
Centralized agent manager for coordinating the multi-agent system.
Enhanced with legal intelligence workflow.
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
        Execute the enhanced workflow with legal intelligence
        PDF → Data Retrieval → Legal Analysis → Synthesis
        """
        logger.info(f"Executing enhanced workflow for query: {query}")
        
        context = AgentContext(query = query, session_id = session_id)
        agent_responses = {}
        errors = []
        
        try:
            # Step 1: Data Retrieval (always needed)
            if "data_retriever" in self.agents:
                data_agent = self.agents["data_retriever"]
                data_result = data_agent.process(context)
                agent_responses["data_retrieval"] = data_result
                
                if not data_result.success:
                    errors.append(f"Data Retrieval failed: {data_result.error_message}")
                    return self._create_error_result(
                        "Data retrieval failed. Please ensure a PDF is uploaded and processed.",
                        agent_responses,
                        errors
                    )
                
                # Add retrieved data to context for next agents
                context.metadata["retrieved_documents"] = data_result.data["retrieved_documents"]
            
            # Step 2: Legal Analysis (for legal queries)
            legal_analysis = None
            if self._is_legal_query(query) and "legal_analyzer" in self.agents:
                legal_agent = self.agents["legal_analyzer"]
                legal_result = legal_agent.process(context)
                agent_responses["legal_analysis"] = legal_result
                
                if legal_result.success:
                    legal_analysis = legal_result.data
                    context.metadata["legal_analysis"] = legal_analysis
                else:
                    errors.append(f"Legal Analysis failed: {legal_result.error_message}")
                    # Continue without legal analysis
            
            # Step 3: Synthesis & Reasoning
            if "summarize_reason" in self.agents:
                synthesize_agent = self.agents["summarize_reason"]
                synthesize_result = synthesize_agent.process(context)
                agent_responses["synthesis"] = synthesize_result
                
                if synthesize_result.success:
                    final_response = synthesize_result.data["synthesized_response"]
                    
                    return WorkflowResult(
                        success = True,
                        final_response = final_response,
                        agent_responses = agent_responses,
                        errors = errors
                    )
                else:
                    errors.append(f"Synthesis failed: {synthesize_result.error_message}")
            
            # Fallback: Use data retrieval results directly
            if "retrieved_documents" in context.metadata:
                final_response = self._create_fallback_response(
                    query, context.metadata["retrieved_documents"]
                )
                
                return WorkflowResult(
                    success = True,
                    final_response = final_response,
                    agent_responses = agent_responses,
                    errors = errors
                )
            else:
                errors.append("No data available for response generation")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            errors.append(f"System error: {str(e)}")
        
        return self._create_error_result(
            "Workflow execution failed. Please try again.",
            agent_responses,
            errors
        )
    
    def _is_legal_query(self, query: str) -> bool:
        """Determine if a query requires legal analysis"""
        legal_keywords = [
            'legal', 'law', 'statute', 'regulation', 'compliance', 'contract',
            'liability', 'rights', 'obligation', 'breach', 'clause', 'article',
            'act', 'code', 'jurisdiction', 'court', 'judge', 'lawsuit',
            'legal advice', 'legal opinion', 'legal issue', 'legal matter'
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in legal_keywords)
    
    def _create_fallback_response(self, query: str, documents: List[Dict]) -> str:
        """Create a fallback response when synthesis fails"""
        response_parts = [
            f"**Response to:** '{query}'",
            "",
            "**Relevant Information Found:**",
            ""
        ]
        
        for doc in documents[:3]:  # Show top 3 documents
            content = doc.get('content', 'No content')
            page = doc.get('page_number', 'N/A')
            response_parts.append(f"**Page {page}:** {content}")
            response_parts.append("")
        
        response_parts.append("*Note: Basic retrieval mode. For detailed legal analysis, rephrase your question.*")
        
        return "\n".join(response_parts)
    
    def _create_error_result(self, error_message: str, agent_responses: Dict, errors: List[str]) -> WorkflowResult:
        """Helper to create error results"""
        return WorkflowResult(
            success = False,
            final_response = error_message,
            agent_responses = agent_responses,
            errors = errors
        )
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get statistics about workflow executions"""
        if not self.workflow_history:
            return {"total_executions": 0, "success_rate": 0.0}
        
        total = len(self.workflow_history)
        successful = sum(1 for result in self.workflow_history if result.success)
        
        # Count agent usage
        agent_usage = {}
        for result in self.workflow_history:
            for agent_name in result.agent_responses.keys():
                agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "agent_usage": agent_usage,
            "last_execution": self.workflow_history[-1] if self.workflow_history else None
        }