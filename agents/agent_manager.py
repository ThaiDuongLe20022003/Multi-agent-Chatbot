"""
Horizontal Multi-Agent Manager with parallel execution and peer-to-peer collaboration.
The orchestrator that enables true horizontal multi-agent architecture.
"""

import logging
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base_agent import BaseAgent, AgentContext, AgentResponse, AgentMessage


logger = logging.getLogger(__name__)


@dataclass
class WorkflowResult:
    """Result of horizontal workflow execution with collaboration tracking"""
    success: bool
    final_response: Any
    agent_responses: Dict[str, AgentResponse]
    errors: List[str]
    processing_time: float = 0.0
    collaboration_log: List[AgentMessage] = None
    workflow_type: str = "horizontal"  # horizontal, sequential, hybrid
    parallel_execution: bool = False
    
    def __post_init__(self):
        if self.collaboration_log is None:
            self.collaboration_log = []


class AgentManager:
    """Orchestrates TRUE horizontal multi-agent workflow with peer collaboration"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflow_history: List[WorkflowResult] = []
        self.response_cache: Dict[str, WorkflowResult] = {}
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent and setup PEER-TO-PEER connections for horizontal architecture"""
        self.agents[agent.name] = agent
        
        # Setup PEER connections for ALL agents - TRUE HORIZONTAL ARCHITECTURE
        for existing_name, existing_agent in self.agents.items():
            if existing_name != agent.name:
                existing_agent.register_peer(agent.name, agent)
                agent.register_peer(existing_name, existing_agent)
        
        logger.info(f"🤝 Registered {agent.name} with peer connections: {list(agent.peer_agents.keys())}")
    
    def execute_workflow(self, query: str, session_id: str = "default") -> WorkflowResult:
        """
        Execute TRUE HORIZONTAL workflow with parallel execution and peer collaboration
        This is the core orchestrator that enables horizontal multi-agent architecture
        """
        start_time = time.time()
        logger.info(f"🚀 Executing HORIZONTAL workflow for: {query}")
        
        context = AgentContext(query = query, session_id = session_id)
        collaboration_log = []
        
        try:
            # STEP 1: Analyze query to dynamically select workflow type
            workflow_type = self._analyze_query_type(query)
            logger.info(f"🎯 Selected workflow type: {workflow_type}")
            
            # STEP 2: Execute appropriate HORIZONTAL workflow
            if workflow_type == "simple_retrieval":
                result = self._execute_simple_retrieval(query, context, start_time, collaboration_log)
            elif workflow_type == "legal_analysis":
                result = self._execute_legal_collaboration(query, context, start_time, collaboration_log)
            elif workflow_type == "comprehensive_analysis":
                result = self._execute_comprehensive_collaboration(query, context, start_time, collaboration_log)
            else:
                result = self._execute_parallel_collaboration(query, context, start_time, collaboration_log)
            
            # Track workflow execution
            self.workflow_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"❌ Horizontal workflow failed: {e}")
            processing_time = time.time() - start_time
            return WorkflowResult(
                success = False,
                final_response = f"Horizontal workflow failed: {str(e)}",
                agent_responses = {},
                errors = [f"System error: {str(e)}"],
                processing_time = processing_time,
                collaboration_log = collaboration_log,
                workflow_type = "horizontal_error"
            )
    
    def _analyze_query_type(self, query: str) -> str:
        """
        Analyze query type DYNAMICALLY for intelligent horizontal routing
        This enables adaptive workflow selection in horizontal architecture
        """
        query_lower = query.lower()
        
        # Legal-focused queries
        legal_indicators = ['legal', 'law', 'statute', 'regulation', 'contract', 'liability', 'rights', 'obligation', 'breach', 'clause']
        # Simple retrieval queries  
        simple_indicators = ['what is', 'summary', 'overview', 'explain', 'tell me about']
        # Complex analysis queries
        complex_indicators = ['analyze', 'compare', 'evaluate', 'implications', 'consequences', 'recommend']
        
        # Calculate scores for each query type
        legal_score = sum(1 for indicator in legal_indicators if indicator in query_lower)
        simple_score = sum(1 for indicator in simple_indicators if indicator in query_lower) 
        complex_score = sum(1 for indicator in complex_indicators if indicator in query_lower)
        
        # Intelligent workflow selection based on query analysis
        if legal_score > 0 and complex_score > 0:
            return "comprehensive_analysis"
        elif legal_score > 0:
            return "legal_analysis"
        elif simple_score > 0 and complex_score == 0:
            return "simple_retrieval"
        else:
            return "parallel_collaboration"
    
    def _execute_parallel_collaboration(self, query: str, context: AgentContext, start_time: float, collaboration_log: List) -> WorkflowResult:
        """
        TRUE HORIZONTAL: Multiple agents work in PARALLEL and COLLABORATE
        This demonstrates the core horizontal architecture with peer-to-peer communication
        """
        agent_responses = {}
        futures = {}
        
        logger.info("🔄 Starting PARALLEL execution with peer collaboration...")
        
        # PARALLEL EXECUTION: Multiple agents run CONCURRENTLY
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit multiple agents for parallel execution
            if "pdf_processor" in self.agents and context.metadata.get("has_pdf"):
                futures['pdf_processing'] = executor.submit(self.agents["pdf_processor"].process_async, context)
            
            if "data_retriever" in self.agents:
                futures['data_retrieval'] = executor.submit(self.agents["data_retriever"].process_async, context)
            
            # Wait for parallel results - TRUE CONCURRENT EXECUTION
            for future in as_completed(futures.values()):
                try:
                    result = future.result()
                    agent_name = [k for k, v in futures.items() if v == future][0]
                    agent_responses[agent_name] = result
                    logger.info(f"✅ {agent_name} completed parallel execution")
                except Exception as e:
                    logger.error(f"❌ Parallel agent failed: {e}")
        
        # PEER-TO-PEER COLLABORATION after parallel execution
        if 'data_retrieval' in agent_responses and agent_responses['data_retrieval'].success:
            context.metadata["retrieved_documents"] = agent_responses['data_retrieval'].data["retrieved_documents"]
            
            # Data Agent broadcasts availability of documents to peers
            data_agent = self.agents.get("data_retriever")
            if data_agent:
                broadcast_responses = data_agent.broadcast_message(
                    "data_available", 
                    {"documents_count": len(context.metadata["retrieved_documents"]), "query": query}
                )
                collaboration_log.extend(broadcast_responses)
        
        # DYNAMIC AGENT ACTIVATION based on query and available data
        final_response = self._orchestrate_dynamic_collaboration(query, context, agent_responses, collaboration_log)
        
        processing_time = time.time() - start_time
        
        return WorkflowResult(
            success = bool(final_response),
            final_response = final_response,
            agent_responses = agent_responses,
            errors = [],
            processing_time = processing_time,
            collaboration_log = collaboration_log,
            workflow_type = "horizontal_parallel",
            parallel_execution = True
        )
    
    def _execute_legal_collaboration(self, query: str, context: AgentContext, start_time: float, collaboration_log: List) -> WorkflowResult:
        """
        HORIZONTAL COLLABORATION for legal queries with intensive peer-to-peer interaction
        Demonstrates specialized workflow in horizontal architecture
        """
        agent_responses = {}
        
        logger.info("⚖️ Starting LEGAL COLLABORATION workflow...")
        
        # Data Retrieval first - foundation for legal analysis
        if "data_retriever" in self.agents:
            data_result = self.agents["data_retriever"].process(context)
            agent_responses["data_retrieval"] = data_result
            
            if data_result.success:
                context.metadata["retrieved_documents"] = data_result.data["retrieved_documents"]
                logger.info(f"📚 Data retrieval found {len(data_result.data['retrieved_documents'])} documents")
        
        # PEER-TO-PEER: Legal Agent requests specific data from Data Agent
        if "legal_analyzer" in self.agents and "data_retriever" in self.agents:
            legal_agent = self.agents["legal_analyzer"]
            data_agent = self.agents["data_retriever"]
            
            # Legal Agent actively requests specific legal information - HORIZONTAL COMMUNICATION
            legal_request = legal_agent.send_message(
                "data_retriever",
                "data_request", 
                {
                    "request_type": "legal_specific", 
                    "legal_topics": self._extract_legal_topics(query),
                    "current_docs": context.metadata.get("retrieved_documents", [])[:2]  # Sample docs
                }
            )
            if legal_request:
                collaboration_log.append(legal_request)
                logger.info(f"📨 Legal → Data: Requested specific legal data")
            
            # Data Agent processes specialized request - PEER RESPONSE
            if legal_request and legal_request.requires_response:
                data_response = data_agent.receive_message(legal_request)
                if data_response:
                    collaboration_log.append(data_response)
                    # Enhance context with legal-specific data from collaboration
                    if "enhanced_results" in data_response.content:
                        context.metadata["legal_enhanced_docs"] = data_response.content["enhanced_results"]
                        logger.info("🔍 Data → Legal: Provided enhanced legal documents")
        
        # Legal Analysis with enriched context from collaboration
        if "legal_analyzer" in self.agents:
            legal_result = self.agents["legal_analyzer"].process(context)
            agent_responses["legal_analysis"] = legal_result
            
            if legal_result.success:
                context.metadata["legal_analysis"] = legal_result.data
                logger.info("✅ Legal analysis completed with collaborative data")
        
        # PEER REVIEW: QA Agent reviews legal analysis
        final_response = self._orchestrate_peer_review(query, context, agent_responses, collaboration_log)
        
        processing_time = time.time() - start_time
        
        return WorkflowResult(
            success = bool(final_response),
            final_response = final_response,
            agent_responses = agent_responses,
            errors = [],
            processing_time = processing_time,
            collaboration_log = collaboration_log,
            workflow_type = "horizontal_legal"
        )
    
    def _orchestrate_dynamic_collaboration(self, query: str, context: AgentContext, agent_responses: Dict, collaboration_log: List) -> str:
        """
        Orchestrate DYNAMIC collaboration between agents based on context
        This is the core intelligence of horizontal architecture
        """
        
        # Determine which agents to activate dynamically based on query and context
        active_agents = self._select_agents_dynamically(query, context)
        logger.info(f"🎯 Dynamically selected agents: {active_agents}")
        
        responses = []
        
        # Execute selected agents with horizontal collaboration
        for agent_name in active_agents:
            if agent_name in self.agents and agent_name not in agent_responses:
                result = self.agents[agent_name].process(context)
                agent_responses[agent_name] = result
                
                if result.success:
                    # PEER COLLABORATION: Share results with other agents
                    self._share_agent_results(agent_name, result, context, collaboration_log)
        
        # COLLABORATIVE SYNTHESIS from multiple agents
        if "summarize_reason" in active_agents and "summarize_reason" in agent_responses:
            synthesis_result = agent_responses["summarize_reason"]
            if synthesis_result.success:
                responses.append(synthesis_result.data.get("synthesized_response", ""))
        
        # Include collaborative insights from horizontal interactions
        collaboration_insights = self._extract_collaboration_insights(collaboration_log)
        if collaboration_insights:
            responses.insert(0, collaboration_insights)
        
        return "\n\n".join(responses) if responses else self._create_fallback_response(query, agent_responses)
    
    def _select_agents_dynamically(self, query: str, context: AgentContext) -> List[str]:
        """
        Dynamically select which agents to activate based on query and context
        Enables adaptive behavior in horizontal architecture
        """
        base_agents = ["data_retriever"]  # Always include data retrieval
        
        query_lower = query.lower()
        
        # Add agents based on query content analysis
        if any(term in query_lower for term in ['legal', 'law', 'contract', 'liability']):
            base_agents.append("legal_analyzer")
        
        if any(term in query_lower for term in ['analyze', 'compare', 'evaluate', 'implications']):
            base_agents.append("summarize_reason")
        
        if any(term in query_lower for term in ['summary', 'overview', 'explain']):
            base_agents.append("summarize_reason")
        
        # Add QA for complex queries in horizontal quality assurance
        if len(query.split()) > 8:  # Longer queries get QA
            base_agents.append("quality_assurance")
        
        return base_agents
    
    def _share_agent_results(self, source_agent: str, result: AgentResponse, context: AgentContext, collaboration_log: List):
        """
        Share agent results with relevant peer agents
        Facilitates information flow in horizontal architecture
        """
        sharing_map = {
            "data_retriever": ["legal_analyzer", "summarize_reason"],
            "legal_analyzer": ["quality_assurance", "summarize_reason"],
            "pdf_processor": ["data_retriever"]
        }
        
        if source_agent in sharing_map:
            for target_agent in sharing_map[source_agent]:
                if target_agent in self.agents:
                    share_message = self.agents[source_agent].send_message(
                        target_agent,
                        "collaboration_request",
                        {
                            "result_type": source_agent,
                            "data_summary": str(result.data)[:500] + "..." if len(str(result.data)) > 500 else str(result.data),
                            "timestamp": self._get_timestamp()
                        },
                        requires_response = False
                    )
                    if share_message:
                        collaboration_log.append(share_message)
    
    def _orchestrate_peer_review(self, query: str, context: AgentContext, agent_responses: Dict, collaboration_log: List) -> str:
        """
        Orchestrate PEER REVIEW process between agents
        Demonstrates quality assurance in horizontal architecture
        """
        
        # Synthesis Agent creates initial response
        if "summarize_reason" in self.agents:
            synthesize_result = self.agents["summarize_reason"].process(context)
            agent_responses["synthesis"] = synthesize_result
            
            if synthesize_result.success:
                draft_response = synthesize_result.data.get("synthesized_response", "")
                
                # PEER REVIEW: QA Agent reviews the draft
                if "quality_assurance" in self.agents:
                    qa_agent = self.agents["quality_assurance"]
                    
                    # QA Agent requests clarification if needed - HORIZONTAL INTERACTION
                    clarification_msg = qa_agent.send_message(
                        "summarize_reason",
                        "clarification_request",
                        {
                            "draft_response": draft_response[:1000],
                            "review_focus": ["completeness", "accuracy", "clarity"],
                            "query_context": query
                        }
                    )
                    if clarification_msg:
                        collaboration_log.append(clarification_msg)
                        logger.info("🔍 QA → Synthesis: Requested clarification")
                    
                    # QA Agent performs review with horizontal context
                    context.metadata["proposed_response"] = draft_response
                    qa_result = qa_agent.process(context)
                    agent_responses["quality_assurance"] = qa_result
                    
                    if qa_result.success and qa_result.data.get("has_improvements", False):
                        improved_response = qa_result.data.get("improved_response", "")
                        logger.info("✅ QA review completed with improvements")
                        return self._format_horizontal_response(query, improved_response, collaboration_log, agent_responses)
                
                return self._format_horizontal_response(query, draft_response, collaboration_log, agent_responses)
        
        return self._create_fallback_response(query, agent_responses)
    
    def _extract_collaboration_insights(self, collaboration_log: List[AgentMessage]) -> str:
        """Extract insights from agent collaborations for horizontal transparency"""
        if not collaboration_log:
            return ""
        
        insights = ["**🤝 Agent Collaboration Insights:**"]
        unique_collaborations = set()
        
        for msg in collaboration_log[-5:]:  # Last 5 collaborations
            if msg.message_type not in ['error']:
                collab_key = f"{msg.sender}→{msg.receiver}:{msg.message_type}"
                if collab_key not in unique_collaborations:
                    insights.append(f"- {msg.sender} → {msg.receiver}: {msg.message_type.replace('_', ' ').title()}")
                    unique_collaborations.add(collab_key)
        
        return "\n".join(insights) + "\n"
    
    def _format_horizontal_response(self, query: str, content: str, collaboration_log: List, agent_responses: Dict) -> str:
        """Format final response with horizontal collaboration context"""
        response_parts = []
        
        # Add collaboration insights from horizontal interactions
        collaboration_insights = self._extract_collaboration_insights(collaboration_log)
        if collaboration_insights:
            response_parts.append(collaboration_insights)
        
        # Add main content from horizontal synthesis
        response_parts.append(f"**🤖 Horizontal Multi-Agent Response:**")
        response_parts.append(content)
        
        # Add horizontal system note
        active_agents = list(agent_responses.keys())
        response_parts.append(f"\n---\n*🔧 Powered by Horizontal AI Agents: {', '.join(active_agents)}*")
        
        return "\n".join(response_parts)
    
    def _extract_legal_topics(self, query: str) -> List[str]:
        """Extract legal topics from query for specialized horizontal routing"""
        topics = []
        query_lower = query.lower()
        
        topic_mapping = {
            'contract': 'contract_law',
            'liability': 'tort_law',
            'rights': 'civil_rights', 
            'obligation': 'contract_law',
            'breach': 'contract_law',
            'clause': 'contract_law',
            'statute': 'statutory_law',
            'regulation': 'administrative_law'
        }
        
        for term, topic in topic_mapping.items():
            if term in query_lower:
                topics.append(topic)
        
        return topics if topics else ['general_law']
    
    def _execute_simple_retrieval(self, query: str, context: AgentContext, start_time: float, collaboration_log: List) -> WorkflowResult:
        """
        Simple workflow for basic queries - horizontal but optimized
        Demonstrates workflow variety in horizontal architecture
        """
        agent_responses = {}
        
        # Parallel execution of PDF + Data retrieval
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            
            if "pdf_processor" in self.agents and context.metadata.get("has_pdf"):
                futures['pdf_processing'] = executor.submit(self.agents["pdf_processor"].process_async, context)
            
            if "data_retriever" in self.agents:
                futures['data_retrieval'] = executor.submit(self.agents["data_retrieval"].process_async, context)
            
            for future in as_completed(futures.values()):
                try:
                    result = future.result()
                    agent_name = [k for k, v in futures.items() if v == future][0]
                    agent_responses[agent_name] = result
                except Exception as e:
                    logger.error(f"Simple retrieval agent failed: {e}")
        
        # Direct to synthesis without complex collaboration for efficiency
        if "summarize_reason" in self.agents:
            if 'data_retrieval' in agent_responses and agent_responses['data_retrieval'].success:
                context.metadata["retrieved_documents"] = agent_responses['data_retrieval'].data["retrieved_documents"]
            
            synthesize_result = self.agents["summarize_reason"].process(context)
            agent_responses["synthesis"] = synthesize_result
            
            if synthesize_result.success:
                final_response = synthesize_result.data.get("synthesized_response", "")
            else:
                final_response = self._create_fallback_response(query, agent_responses)
        else:
            final_response = self._create_fallback_response(query, agent_responses)
        
        processing_time = time.time() - start_time
        
        return WorkflowResult(
            success = bool(final_response),
            final_response = final_response,
            agent_responses = agent_responses,
            errors = [],
            processing_time = processing_time,
            collaboration_log = collaboration_log,
            workflow_type = "horizontal_simple",
            parallel_execution = True
        )
    
    def _execute_comprehensive_collaboration(self, query: str, context: AgentContext, start_time: float, collaboration_log: List) -> WorkflowResult:
        """
        Comprehensive workflow with maximum horizontal collaboration
        Demonstrates full capabilities of horizontal architecture
        """
        # Start with parallel collaboration foundation
        parallel_result = self._execute_parallel_collaboration(query, context, start_time, collaboration_log)
        
        # Then add intensive peer review for quality assurance
        if parallel_result.success and "summarize_reason" in self.agents:
            context.metadata["initial_response"] = parallel_result.final_response
            
            # Multi-agent peer review in horizontal architecture
            review_response = self._orchestrate_multi_agent_review(query, context, parallel_result.agent_responses, collaboration_log)
            if review_response and review_response != parallel_result.final_response:
                parallel_result.final_response = review_response
        
        return parallel_result
    
    def _orchestrate_multi_agent_review(self, query: str, context: AgentContext, agent_responses: Dict, collaboration_log: List) -> str:
        """
        Orchestrate review by multiple agents in horizontal quality assurance
        """
        draft_response = context.metadata.get("initial_response", "")
        
        if not draft_response:
            return draft_response
        
        # Legal Agent review for specialized validation
        if "legal_analyzer" in self.agents:
            legal_review = self.agents["legal_analyzer"].send_message(
                "summarize_reason",
                "validation_request",
                {"content": draft_response, "validation_type": "legal_accuracy"}
            )
            if legal_review:
                collaboration_log.append(legal_review)
        
        # QA Agent comprehensive review in horizontal context
        if "quality_assurance" in self.agents:
            context.metadata["proposed_response"] = draft_response
            qa_result = self.agents["quality_assurance"].process(context)
            if qa_result.success and qa_result.data.get("has_improvements", False):
                return qa_result.data.get("improved_response", draft_response)
        
        return draft_response
    
    def _create_fallback_response(self, query: str, agent_responses: Dict) -> str:
        """Create fallback response when horizontal collaboration has limited results"""
        response_parts = [f"**Response to:** '{query}'", ""]
        
        if 'data_retrieval' in agent_responses and agent_responses['data_retrieval'].success:
            docs = agent_responses['data_retrieval'].data.get('retrieved_documents', [])
            response_parts.append("**Relevant Information:**")
            for doc in docs[:2]:
                content = doc.get('content', '')[:300] + '...' if len(doc.get('content', '')) > 300 else doc.get('content', '')
                response_parts.append(f"- {content}")
        else:
            response_parts.append("No relevant information found.")
        
        response_parts.append("\n*💡 Horizontal AI System: Basic retrieval mode*")
        
        return "\n".join(response_parts)
    
    def _get_timestamp(self):
        """Get current timestamp for collaboration tracking"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_workflow_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about horizontal workflow executions"""
        if not self.workflow_history:
            return {"total_executions": 0, "success_rate": 0.0}
        
        total = len(self.workflow_history)
        successful = sum(1 for result in self.workflow_history if result.success)
        parallel_count = sum(1 for result in self.workflow_history if result.parallel_execution)
        
        # Calculate average processing time
        avg_time = sum(result.processing_time for result in self.workflow_history) / total if total > 0 else 0
        
        # Count agent usage and collaborations
        agent_usage = {}
        collaboration_count = 0
        
        for result in self.workflow_history:
            for agent_name in result.agent_responses.keys():
                agent_usage[agent_name] = agent_usage.get(agent_name, 0) + 1
            collaboration_count += len(result.collaboration_log)
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_processing_time": round(avg_time, 2),
            "parallel_executions": parallel_count,
            "total_collaborations": collaboration_count,
            "agent_usage": agent_usage,
            "workflow_types": [result.workflow_type for result in self.workflow_history[-5:]] if self.workflow_history else []
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        cache_size = len(self.response_cache)
        self.response_cache.clear()
        logger.info(f"Cleared cache with {cache_size} entries")
        return cache_size