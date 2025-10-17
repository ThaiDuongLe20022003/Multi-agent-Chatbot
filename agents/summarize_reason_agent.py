"""
Summarize & Reason Agent with horizontal collaboration capabilities.
Enhanced for multi-source synthesis and collaborative response generation.
"""

import logging
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentContext, AgentResponse


logger = logging.getLogger(__name__)


class SummarizeReasonAgent(BaseAgent):
    """Agent specialized in synthesis, summarization, and logical reasoning with horizontal collaboration"""
    
    def __init__(self):
        super().__init__("summarize_reason")
        self.llm = None  # Will be initialized with selected model
        
        # Synthesis prompt template
        self.synthesis_prompt = ChatPromptTemplate.from_template("""
        You are an expert at synthesizing information and providing clear, logical reasoning.

        ORIGINAL QUERY:
        {query}

        LEGAL ANALYSIS:
        {legal_analysis}

        SUPPORTING DOCUMENTS:
        {supporting_docs}

        Your task is to synthesize this information into a comprehensive, well-reasoned response:

        1. **Executive Summary**: Begin with a concise summary of the key findings.
        2. **Logical Flow**: Present information in a logical, easy-to-follow structure.
        3. **Key Points**: Highlight the most important legal points and insights.
        4. **Practical Implications**: Explain what this means in practical terms.
        5. **Confidence Level**: Indicate the confidence level based on available information.

        Ensure the response is professional, clear, and directly addresses the original query.

        SYNTHESIZED RESPONSE:
        """)
    
    def initialize_model(self, model_name: str):
        """Initialize the LLM with the selected model"""
        try:
            self.llm = ChatOllama(model = model_name, temperature = 0.1, request_timeout = 120.0)
            logger.info(f"Summarize reason agent initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize summarize reason agent with model {model_name}: {e}")
            raise
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Synthesize and reason across multiple information sources with horizontal collaboration"""
        start_time = self._get_timestamp()
        collaborations = []
        
        try:
            if not self.llm:
                return self._create_error_response("Summarize reason agent not initialized with a model")
                
            query = context.query
            legal_analysis = context.metadata.get("legal_analysis")
            retrieved_docs = context.metadata.get("retrieved_documents", [])
            
            # COLLECT INPUTS FROM MULTIPLE SOURCES through horizontal collaboration
            additional_inputs = self._gather_additional_inputs(query, context, collaborations)
            
            if not legal_analysis and not retrieved_docs and not additional_inputs:
                return self._create_error_response("No analysis or documents available for synthesis")
            
            # Prepare supporting documents context
            supporting_context = self._prepare_supporting_context(retrieved_docs, additional_inputs)
            
            # Synthesize comprehensive response
            synthesized_response = self._synthesize_response(
                query, legal_analysis, supporting_context
            )
            
            # REQUEST PEER REVIEW from QA Agent
            peer_review_request = self.send_message(
                "quality_assurance",
                "validation_request",
                {
                    "content_type": "synthesized_response",
                    "content": synthesized_response[:1000],  # Sample for review
                    "query": query,
                    "validation_focus": ["clarity", "completeness", "accuracy"]
                }
            )
            if peer_review_request:
                collaborations.append(peer_review_request)
                logger.info("📨 Summarize → QA: Requested peer review")
            
            result_data = {
                "original_query": query,
                "synthesized_response": synthesized_response,
                "sources_used": {
                    "legal_analysis": bool(legal_analysis),
                    "supporting_documents": len(retrieved_docs),
                    "additional_inputs": len(additional_inputs),
                    "peer_review_requested": bool(peer_review_request)
                },
                "response_type": "comprehensive_synthesis",
                "collaboration_activities": len(collaborations)
            }
            
            processing_time = self._calculate_processing_time(start_time)
            logger.info(f"✅ Synthesis completed: {len(synthesized_response)} chars, {len(collaborations)} collaborations")
            
            return self._create_success_response(
                data = result_data,
                collaborations = collaborations,
                processing_time = processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Synthesis error: {e}")
            return self._create_error_response(f"Synthesis failed: {str(e)}")
    
    def _handle_data_request(self, content: Dict) -> Dict:
        """
        Handle data requests from peer agents
        Example: QA Agent requests synthesis methodology
        """
        try:
            request_type = content.get("request_type", "synthesis_info")
            
            if request_type == "synthesis_methodology":
                # Provide information about synthesis methodology
                return {
                    "status": "success",
                    "request_type": "synthesis_methodology",
                    "methodology": {
                        "approach": "multi_source_synthesis",
                        "techniques": ["executive_summarization", "logical_structuring", "key_point_extraction"],
                        "quality_focus": ["clarity", "completeness", "practical_relevance"]
                    },
                    "input_sources": ["legal_analysis", "retrieved_documents", "collaborative_inputs"]
                }
            
            elif request_type == "reasoning_chain":
                # Provide reasoning chain for transparency
                synthesis_content = content.get("synthesized_content", "")
                reasoning_chain = self._extract_reasoning_chain(synthesis_content)
                
                return {
                    "status": "success",
                    "request_type": "reasoning_chain",
                    "reasoning_steps": reasoning_chain,
                    "reasoning_quality": "structured" if len(reasoning_chain) > 2 else "basic"
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Synthesis data available",
                    "available_data": ["synthesis_methodology", "reasoning_chain", "source_integration"]
                }
                
        except Exception as e:
            logger.error(f"Error handling data request: {e}")
            return {"status": "error", "message": f"Data request failed: {str(e)}"}
    
    def _handle_analysis_request(self, content: Dict) -> Dict:
        """
        Handle analysis requests from peer agents
        Example: Legal Agent requests synthesis quality analysis
        """
        try:
            request_type = content.get("request_type", "synthesis_quality")
            synthesized_content = content.get("content", "")
            original_sources = content.get("sources", {})
            
            if request_type == "synthesis_quality":
                # Analyze synthesis quality
                quality_metrics = self._analyze_synthesis_quality(synthesized_content, original_sources)
                
                return {
                    "status": "success",
                    "request_type": "synthesis_quality",
                    "quality_metrics": quality_metrics,
                    "overall_score": quality_metrics.get("overall_score", 0),
                    "improvement_suggestions": quality_metrics.get("suggestions", [])
                }
            
            elif request_type == "coverage_analysis":
                # Analyze how well sources are covered
                coverage_analysis = self._analyze_source_coverage(synthesized_content, original_sources)
                
                return {
                    "status": "success",
                    "request_type": "coverage_analysis",
                    "coverage_score": coverage_analysis.get("coverage_score", 0),
                    "missing_elements": coverage_analysis.get("missing_elements", []),
                    "coverage_completeness": coverage_analysis.get("completeness", "partial")
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Synthesis analysis services available",
                    "available_analyses": ["synthesis_quality", "coverage_analysis", "reasoning_analysis"]
                }
                
        except Exception as e:
            logger.error(f"Error handling analysis request: {e}")
            return {"status": "error", "message": f"Analysis request failed: {str(e)}"}
    
    def _handle_clarification(self, content: Dict) -> Dict:
        """
        Handle clarification requests from peer agents
        Example: QA Agent requests clarification on reasoning
        """
        try:
            clarification_type = content.get("clarification_type", "reasoning")
            content_to_clarify = content.get("content", "")
            specific_questions = content.get("questions", [])
            
            if clarification_type == "reasoning":
                # Provide clarification on reasoning
                clarification = self._clarify_reasoning(content_to_clarify, specific_questions)
                
                return {
                    "status": "success",
                    "clarification_type": "reasoning",
                    "clarification_provided": clarification,
                    "questions_addressed": specific_questions if specific_questions else ["general reasoning explanation"]
                }
            
            elif clarification_type == "source_integration":
                # Clarify how sources were integrated
                integration_explanation = self._explain_source_integration(content_to_clarify)
                
                return {
                    "status": "success",
                    "clarification_type": "source_integration",
                    "integration_explanation": integration_explanation,
                    "integration_strategy": "multi_source_synthesis"
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Clarification services available",
                    "available_clarifications": ["reasoning", "source_integration", "synthesis_methodology"]
                }
                
        except Exception as e:
            logger.error(f"Error handling clarification request: {e}")
            return {"status": "error", "message": f"Clarification failed: {str(e)}"}
    
    def _handle_collaboration_request(self, content: Dict) -> Dict:
        """
        Handle collaboration requests from peer agents
        Example: Data Agent requests synthesis of multiple retrievals
        """
        try:
            collaboration_type = content.get("collaboration_type", "content_synthesis")
            content_to_synthesize = content.get("content", "")
            synthesis_guidelines = content.get("guidelines", {})
            
            if collaboration_type == "content_synthesis":
                # Perform synthesis for collaborative request
                synthesized_content = self._perform_targeted_synthesis(content_to_synthesize, synthesis_guidelines)
                
                return {
                    "status": "success",
                    "collaboration_type": "content_synthesis",
                    "synthesized_content": synthesized_content,
                    "synthesis_approach": "guideline_driven",
                    "content_length": len(synthesized_content)
                }
            
            elif collaboration_type == "summary_generation":
                # Generate summary for collaborative purposes
                summary = self._generate_collaborative_summary(content_to_synthesize, synthesis_guidelines)
                
                return {
                    "status": "success",
                    "collaboration_type": "summary_generation",
                    "generated_summary": summary,
                    "summary_type": synthesis_guidelines.get("summary_type", "executive")
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Collaborative synthesis services available",
                    "available_collaborations": ["content_synthesis", "summary_generation", "reasoning_assistance"]
                }
                
        except Exception as e:
            logger.error(f"Error handling collaboration request: {e}")
            return {"status": "error", "message": f"Collaboration failed: {str(e)}"}
    
    def _gather_additional_inputs(self, query: str, context: AgentContext, collaborations: List) -> Dict[str, Any]:
        """Gather additional inputs through horizontal collaboration"""
        additional_inputs = {}
        
        # Request legal perspective if not already available
        if not context.metadata.get("legal_analysis") and "legal_analyzer" in self.peer_agents:
            legal_request = self.send_message(
                "legal_analyzer",
                "analysis_request",
                {
                    "request_type": "legal_perspective",
                    "content": context.metadata.get("retrieved_documents", [])[:2],
                    "focus_areas": self._extract_query_focus_areas(query)
                }
            )
            if legal_request and "legal_perspective" in legal_request.content:
                additional_inputs["legal_perspective"] = legal_request.content["legal_perspective"]
                collaborations.append(legal_request)
        
        return additional_inputs
    
    def _prepare_supporting_context(self, retrieved_documents: List[Dict[str, Any]], additional_inputs: Dict) -> str:
        """Prepare supporting context from multiple sources"""
        context_parts = ["SUPPORTING INFORMATION:"]
        
        # Add retrieved documents
        if retrieved_documents:
            context_parts.append("RETRIEVED DOCUMENTS:")
            for i, doc in enumerate(retrieved_documents[:3]):
                content_preview = doc.get('content', '')[:300]
                page_info = f"Page {doc.get('page_number', 'N/A')}"
                context_parts.append(f"{i+1}. [{page_info}] {content_preview}...")
        
        # Add additional inputs from collaborations
        if additional_inputs:
            context_parts.append("COLLABORATIVE INPUTS:")
            for source, content in additional_inputs.items():
                context_parts.append(f"- {source.replace('_', ' ').title()}: {content[:200]}...")
        
        return "\n".join(context_parts) if context_parts else "No supporting documents available."
    
    def _synthesize_response(self, query: str, legal_analysis: str, supporting_docs: str) -> str:
        """Synthesize comprehensive response using LLM"""
        try:
            # If legal_analysis is a dictionary, extract the text
            if isinstance(legal_analysis, dict):
                legal_text = legal_analysis.get('legal_analysis', str(legal_analysis))
            else:
                legal_text = legal_analysis
            
            prompt = self.synthesis_prompt.format(
                query = query,
                legal_analysis = legal_text[:2000] if legal_text else "No legal analysis available",
                supporting_docs = supporting_docs
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM synthesis error: {e}")
            return f"Synthesis could not be completed due to an error: {str(e)}"
    
    def _extract_query_focus_areas(self, query: str) -> List[str]:
        """Extract focus areas from query for targeted collaboration"""
        focus_areas = []
        query_lower = query.lower()
        
        if any(term in query_lower for term in ['legal', 'law', 'contract']):
            focus_areas.append('legal_analysis')
        if any(term in query_lower for term in ['summary', 'overview']):
            focus_areas.append('summarization')
        if any(term in query_lower for term in ['analyze', 'evaluate']):
            focus_areas.append('critical_analysis')
        
        return focus_areas if focus_areas else ['general_synthesis']
    
    def _extract_reasoning_chain(self, content: str) -> List[str]:
        """Extract reasoning chain from synthesized content"""
        # Simple extraction based on reasoning indicators
        reasoning_indicators = ['because', 'therefore', 'thus', 'accordingly', 'consequently', 'as a result']
        sentences = content.split('. ')
        reasoning_steps = [sentence for sentence in sentences if any(indicator in sentence.lower() for indicator in reasoning_indicators)]
        return reasoning_steps[:5]  # Return top 5 reasoning steps
    
    def _analyze_synthesis_quality(self, synthesized_content: str, original_sources: Dict) -> Dict[str, Any]:
        """Analyze synthesis quality for horizontal quality assurance"""
        content_length = len(synthesized_content)
        has_structure = any(marker in synthesized_content for marker in ['**', '-', '1.', '•'])
        has_reasoning = any(indicator in synthesized_content.lower() for indicator in ['because', 'therefore', 'thus'])
        
        return {
            "overall_score": min(10, (content_length / 100 + (5 if has_structure else 0) + (3 if has_reasoning else 0))),
            "aspects": {
                "content_completeness": min(10, content_length / 50),
                "structural_quality": 8 if has_structure else 4,
                "reasoning_presence": 7 if has_reasoning else 3
            },
            "suggestions": [
                "Add more structural markers" if not has_structure else "Structure is good",
                "Include explicit reasoning" if not has_reasoning else "Reasoning is present"
            ]
        }
    
    def _analyze_source_coverage(self, synthesized_content: str, original_sources: Dict) -> Dict[str, Any]:
        """Analyze how well original sources are covered in synthesis"""
        # Simplified coverage analysis
        source_count = original_sources.get('document_count', 0)
        coverage_score = min(10, source_count * 2)  # Simple scoring
        
        return {
            "coverage_score": coverage_score,
            "missing_elements": ["Detailed analysis"] if coverage_score < 7 else [],
            "completeness": "comprehensive" if coverage_score >= 8 else "adequate" if coverage_score >= 5 else "partial"
        }
    
    def _clarify_reasoning(self, content: str, questions: List[str]) -> str:
        """Provide clarification on reasoning in content"""
        try:
            questions_text = "\n".join([f"- {q}" for q in questions]) if questions else "general reasoning approach"
            prompt = f"""
            Provide clarification about the reasoning in the following content.
            Focus on these questions: {questions_text}
            
            CONTENT:
            {content[:1000]}
            
            CLARIFICATION:
            """
            
            if self.llm:
                response = self.llm.invoke(prompt)
                return response.content.strip()
            else:
                return "Clarification unavailable - agent not initialized"
                
        except Exception as e:
            logger.error(f"Error providing clarification: {e}")
            return f"Clarification failed: {str(e)}"
    
    def _explain_source_integration(self, content: str) -> str:
        """Explain how sources were integrated into the synthesis"""
        return f"The synthesis integrates multiple sources by extracting key insights and combining them into a coherent narrative. Sources are weighted based on relevance to the query and integrated to provide comprehensive coverage of the topic."
    
    def _perform_targeted_synthesis(self, content: str, guidelines: Dict) -> str:
        """Perform targeted synthesis based on specific guidelines"""
        try:
            focus_areas = guidelines.get('focus_areas', ['key_points'])
            synthesis_type = guidelines.get('synthesis_type', 'summary')
            
            prompt = f"""
            Create a {synthesis_type} of the following content, focusing on: {', '.join(focus_areas)}
            
            CONTENT:
            {content[:1500]}
            
            {synthesis_type.upper()}:
            """
            
            if self.llm:
                response = self.llm.invoke(prompt)
                return response.content.strip()
            else:
                return "Targeted synthesis unavailable"
                
        except Exception as e:
            logger.error(f"Error performing targeted synthesis: {e}")
            return f"Targeted synthesis failed: {str(e)}"
    
    def _generate_collaborative_summary(self, content: str, guidelines: Dict) -> str:
        """Generate summary for collaborative purposes"""
        summary_type = guidelines.get('summary_type', 'executive')
        return self._perform_targeted_synthesis(content, {'synthesis_type': summary_type})
    
    def _calculate_processing_time(self, start_time: str) -> float:
        """Calculate processing time for performance tracking"""
        from datetime import datetime
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.now()
        return (end_dt - start_dt).total_seconds()
    
    def get_agent_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities for horizontal system discovery"""
        return {
            "agent_name": self.name,
            "role": "Information synthesis and reasoning specialist",
            "capabilities": [
                "multi_source_synthesis",
                "executive_summarization", 
                "logical_reasoning",
                "collaborative_response_generation"
            ],
            "horizontal_features": [
                "peer_input_gathering",
                "collaborative_synthesis",
                "reasoning_clarification",
                "quality_peer_review"
            ],
            "synthesis_strategies": [
                "executive_summary",
                "structured_analysis",
                "key_point_extraction",
                "practical_implications"
            ]
        }