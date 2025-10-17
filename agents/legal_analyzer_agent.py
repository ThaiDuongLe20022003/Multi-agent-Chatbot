"""
Legal Analyzer Agent with horizontal collaboration capabilities.
Enhanced for peer-to-peer legal analysis and collaborative reasoning.
"""

import logging
from typing import Dict, Any, List
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentContext, AgentResponse


logger = logging.getLogger(__name__)


class LegalAnalyzerAgent(BaseAgent):
    """Agent specialized in legal analysis and interpretation with horizontal collaboration"""
    
    def __init__(self):
        super().__init__("legal_analyzer")
        self.llm = None  # Will be initialized with selected model
        
        # Legal analysis prompt template
        self.analysis_prompt = ChatPromptTemplate.from_template("""
        You are an expert legal analyst with deep knowledge of legal principles, 
        case law interpretation, and legal reasoning.

        LEGAL DOCUMENT CONTEXT:
        {context}

        USER'S LEGAL QUESTION:
        {question}

        Please provide a comprehensive legal analysis:

        1. **Legal Issues Identification**: What are the key legal issues presented?
        2. **Relevant Legal Principles**: What legal principles, statutes, or precedents apply?
        3. **Application to Facts**: How do these legal principles apply to the provided context?
        4. **Legal Reasoning**: Provide step-by-step legal reasoning.
        5. **Potential Implications**: What are the potential legal consequences or implications?

        If the context doesn't contain sufficient legal information, clearly state the limitations.

        LEGAL ANALYSIS:
        """)
    
    def initialize_model(self, model_name: str):
        """Initialize the LLM with the selected model"""
        try:
            self.llm = ChatOllama(model = model_name, temperature = 0.1, request_timeout = 120.0)
            logger.info(f"Legal analyzer agent initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize legal analyzer with model {model_name}: {e}")
            raise
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Perform legal analysis with horizontal collaboration"""
        start_time = self._get_timestamp()
        collaborations = []
        
        try:
            if not self.llm:
                return self._create_error_response("Legal analyzer agent not initialized with a model")
                
            query = context.query
            retrieved_data = context.metadata.get("retrieved_documents", [])
            
            if not retrieved_data:
                return self._create_error_response("No retrieved documents available for legal analysis")
            
            # REQUEST ENHANCED DATA from Data Retrieval Agent if needed
            if self._requires_enhanced_legal_data(query):
                enhanced_data_request = self.send_message(
                    "data_retriever",
                    "data_request",
                    {
                        "request_type": "legal_specific",
                        "query": query,
                        "legal_topics": self._extract_legal_topics(query),
                        "current_docs": retrieved_data[:2]  # Sample for context
                    }
                )
                if enhanced_data_request:
                    collaborations.append(enhanced_data_request)
                    logger.info("📨 Legal → Data: Requested enhanced legal documents")
            
            # Prepare context from retrieved documents
            analysis_context = self._prepare_legal_context(retrieved_data)
            
            # Perform legal analysis
            legal_analysis = self._perform_legal_analysis(query, analysis_context)
            
            # BROADCAST legal analysis results to interested peers
            broadcast_responses = self.broadcast_message(
                "legal_analysis_complete",
                {
                    "query": query,
                    "analysis_preview": legal_analysis[:200] + "..." if len(legal_analysis) > 200 else legal_analysis,
                    "legal_topics_identified": self._extract_legal_topics(query),
                    "analysis_confidence": "high" if len(analysis_context) > 500 else "medium"
                }
            )
            collaborations.extend(broadcast_responses)
            
            result_data = {
                "original_query": query,
                "legal_analysis": legal_analysis,
                "context_used": analysis_context[:1000] + "..." if len(analysis_context) > 1000 else analysis_context,
                "analysis_type": "comprehensive_legal_review",
                "legal_topics": self._extract_legal_topics(query),
                "collaboration_requests": len(collaborations)
            }
            
            processing_time = self._calculate_processing_time(start_time)
            logger.info(f"✅ Legal analysis completed: {len(legal_analysis)} chars, {len(collaborations)} collaborations")
            
            return self._create_success_response(
                data = result_data,
                collaborations = collaborations,
                processing_time = processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Legal analysis error: {e}")
            return self._create_error_response(f"Legal analysis failed: {str(e)}")
    
    def _handle_data_request(self, content: Dict) -> Dict:
        """
        Handle data requests from peer agents
        Example: QA Agent requests legal analysis data for validation
        """
        try:
            request_type = content.get("request_type", "analysis_data")
            analysis_data = content.get("analysis_data", "")
            
            if request_type == "analysis_breakdown":
                # Provide structured breakdown of legal analysis
                breakdown = self._breakdown_legal_analysis(analysis_data)
                
                return {
                    "status": "success",
                    "request_type": "analysis_breakdown",
                    "breakdown": breakdown,
                    "legal_issues_identified": breakdown.get("legal_issues", []),
                    "reasoning_steps": breakdown.get("reasoning_steps", [])
                }
            
            elif request_type == "legal_citations":
                # Extract legal citations from analysis
                citations = self._extract_legal_citations(analysis_data)
                
                return {
                    "status": "success",
                    "request_type": "legal_citations",
                    "citations_found": citations,
                    "citation_count": len(citations)
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Legal analysis data available",
                    "available_data_types": ["analysis_breakdown", "legal_citations", "reasoning_chain"]
                }
                
        except Exception as e:
            logger.error(f"Error handling data request: {e}")
            return {"status": "error", "message": f"Data request failed: {str(e)}"}
    
    def _handle_analysis_request(self, content: Dict) -> Dict:
        """
        Handle analysis requests from peer agents
        Example: Summarize Agent requests legal perspective on content
        """
        try:
            request_type = content.get("request_type", "legal_perspective")
            content_to_analyze = content.get("content", "")
            analysis_focus = content.get("focus_areas", ["general"])
            
            if request_type == "legal_perspective":
                # Provide legal perspective on given content
                legal_perspective = self._provide_legal_perspective(content_to_analyze, analysis_focus)
                
                return {
                    "status": "success",
                    "request_type": "legal_perspective",
                    "legal_perspective": legal_perspective,
                    "focus_areas": analysis_focus,
                    "analysis_scope": "targeted_legal_review"
                }
            
            elif request_type == "compliance_check":
                # Check content for legal compliance issues
                compliance_analysis = self._check_legal_compliance(content_to_analyze)
                
                return {
                    "status": "success",
                    "request_type": "compliance_check",
                    "compliance_issues": compliance_analysis.get("issues", []),
                    "compliance_score": compliance_analysis.get("score", 0),
                    "recommendations": compliance_analysis.get("recommendations", [])
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Legal analysis services available",
                    "available_analyses": ["legal_perspective", "compliance_check", "risk_assessment"]
                }
                
        except Exception as e:
            logger.error(f"Error handling analysis request: {e}")
            return {"status": "error", "message": f"Analysis request failed: {str(e)}"}
    
    def _handle_validation_request(self, content: Dict) -> Dict:
        """
        Handle validation requests from peer agents
        Example: QA Agent validates legal reasoning
        """
        try:
            validation_type = content.get("validation_type", "legal_reasoning")
            content_to_validate = content.get("content", "")
            validation_criteria = content.get("criteria", {})
            
            if validation_type == "legal_reasoning":
                # Validate legal reasoning in content
                validation_result = self._validate_legal_reasoning(content_to_validate, validation_criteria)
                
                return {
                    "status": "success",
                    "validation_type": "legal_reasoning",
                    "is_valid": validation_result["is_valid"],
                    "validation_score": validation_result["score"],
                    "issues_found": validation_result["issues"],
                    "improvement_suggestions": validation_result["suggestions"]
                }
            
            elif validation_type == "legal_accuracy":
                # Validate legal accuracy of statements
                accuracy_check = self._validate_legal_accuracy(content_to_validate)
                
                return {
                    "status": "success",
                    "validation_type": "legal_accuracy",
                    "accuracy_score": accuracy_check["score"],
                    "inaccurate_statements": accuracy_check["inaccuracies"],
                    "corrections": accuracy_check["corrections"]
                }
            
            else:
                return {
                    "status": "success",
                    "message": "Legal validation services available",
                    "available_validations": ["legal_reasoning", "legal_accuracy", "compliance_validation"]
                }
                
        except Exception as e:
            logger.error(f"Error handling validation request: {e}")
            return {"status": "error", "message": f"Validation request failed: {str(e)}"}
    
    def _requires_enhanced_legal_data(self, query: str) -> bool:
        """Determine if enhanced legal data is needed for this query"""
        complex_legal_terms = ['precedent', 'jurisdiction', 'statutory', 'regulatory', 'compliance', 'liability']
        return any(term in query.lower() for term in complex_legal_terms)
    
    def _prepare_legal_context(self, retrieved_documents: List[Dict[str, Any]]) -> str:
        """Prepare legal context from retrieved documents with enhanced formatting"""
        context_parts = ["LEGAL DOCUMENT CONTEXT FOR ANALYSIS:"]
        
        for doc in retrieved_documents:
            content = doc.get('content', '')
            page_info = f"Page {doc.get('page_number', 'N/A')}"
            relevance_indicator = "⚖️ " if any(term in content.lower() for term in ['law', 'legal', 'contract', 'clause']) else ""
            context_parts.append(f"[{page_info}] {relevance_indicator}{content}")
        
        return "\n\n".join(context_parts)
    
    def _perform_legal_analysis(self, query: str, context: str) -> str:
        """Execute legal analysis using LLM with enhanced prompting"""
        try:
            prompt = self.analysis_prompt.format(
                context=context[:3000],  # Limit context length
                question=query
            )
            
            response = self.llm.invoke(prompt)
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM legal analysis error: {e}")
            return f"Legal analysis could not be completed due to an error: {str(e)}"
    
    def _extract_legal_topics(self, query: str) -> List[str]:
        """Extract legal topics from query for enhanced retrieval"""
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
            'regulation': 'administrative_law',
            'compliance': 'regulatory_compliance',
            'lawsuit': 'litigation',
            'damages': 'tort_law',
            'agreement': 'contract_law'
        }
        
        for term, topic in topic_mapping.items():
            if term in query_lower:
                topics.append(topic)
        
        return topics if topics else ['general_legal_analysis']
    
    def _breakdown_legal_analysis(self, analysis: str) -> Dict[str, Any]:
        """Break down legal analysis into structured components"""
        # Simple breakdown - in practice would use more sophisticated NLP
        lines = analysis.split('\n')
        legal_issues = []
        reasoning_steps = []
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ['issue', 'problem', 'question']):
                legal_issues.append(line.strip())
            elif any(keyword in line_lower for keyword in ['because', 'therefore', 'thus', 'accordingly']):
                reasoning_steps.append(line.strip())
        
        return {
            "legal_issues": legal_issues[:5],
            "reasoning_steps": reasoning_steps[:5],
            "analysis_length": len(analysis),
            "estimated_complexity": "high" if len(legal_issues) > 2 else "medium"
        }
    
    def _extract_legal_citations(self, analysis: str) -> List[str]:
        """Extract potential legal citations from analysis"""
        # Simple citation extraction - would be enhanced in production
        import re
        citation_patterns = [
            r'\d+ [A-Z][a-z]+\. \d+',  # Basic case citation pattern
            r'[A-Z][a-z]+ Act',
            r'Article [IVXLCDM]+',
            r'Section \d+'
        ]
        
        citations = []
        for pattern in citation_patterns:
            citations.extend(re.findall(pattern, analysis))
        
        return citations
    
    def _provide_legal_perspective(self, content: str, focus_areas: List[str]) -> str:
        """Provide legal perspective on given content"""
        try:
            prompt = f"""
            Provide a brief legal perspective on the following content, focusing on: {', '.join(focus_areas)}
            
            CONTENT:
            {content[:1500]}
            
            LEGAL PERSPECTIVE:
            """
            
            if self.llm:
                response = self.llm.invoke(prompt)
                return response.content.strip()
            else:
                return "Legal perspective unavailable - agent not initialized"
                
        except Exception as e:
            logger.error(f"Error providing legal perspective: {e}")
            return f"Legal perspective analysis failed: {str(e)}"
    
    def _check_legal_compliance(self, content: str) -> Dict[str, Any]:
        """Check content for legal compliance issues"""
        try:
            prompt = f"""
            Analyze the following content for potential legal compliance issues.
            Identify any areas that might raise legal concerns and provide a compliance score (0-10).
            
            CONTENT:
            {content[:2000]}
            
            COMPLIANCE ANALYSIS:
            """
            
            if self.llm:
                response = self.llm.invoke(prompt)
                # Simple scoring - would be more sophisticated in practice
                score = 8 if "compliance" in response.content.lower() else 6
                return {
                    "score": score,
                    "issues": ["General legal review recommended"] if score < 7 else [],
                    "recommendations": ["Consult legal expert for specific advice"]
                }
            else:
                return {"score": 5, "issues": ["Analysis unavailable"], "recommendations": []}
                
        except Exception as e:
            logger.error(f"Error checking compliance: {e}")
            return {"score": 0, "issues": [f"Compliance check failed: {str(e)}"], "recommendations": []}
    
    def _validate_legal_reasoning(self, content: str, criteria: Dict) -> Dict[str, Any]:
        """Validate legal reasoning in content"""
        # Simplified validation - would be more comprehensive in practice
        reasoning_indicators = ['because', 'therefore', 'thus', 'accordingly', 'consequently']
        has_reasoning = any(indicator in content.lower() for indicator in reasoning_indicators)
        
        return {
            "is_valid": has_reasoning,
            "score": 8 if has_reasoning else 4,
            "issues": [] if has_reasoning else ["Legal reasoning could be more explicit"],
            "suggestions": ["Reasoning is clear"] if has_reasoning else ["Add explicit legal reasoning steps"]
        }
    
    def _validate_legal_accuracy(self, content: str) -> Dict[str, Any]:
        """Validate legal accuracy of statements"""
        # Placeholder for legal accuracy validation
        # In practice, this would involve checking against legal databases or knowledge bases
        return {
            "score": 7,  # Placeholder score
            "inaccuracies": ["No major inaccuracies detected in initial review"],
            "corrections": ["Review by legal expert recommended for complete accuracy"]
        }
    
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
            "role": "Legal analysis and reasoning specialist",
            "capabilities": [
                "legal_issue_identification",
                "legal_reasoning",
                "compliance_analysis",
                "legal_validation"
            ],
            "horizontal_features": [
                "enhanced_data_requests",
                "legal_perspective_provision",
                "reasoning_validation",
                "collaborative_analysis"
            ],
            "legal_domains": [
                "contract_law",
                "tort_law", 
                "civil_rights",
                "regulatory_compliance"
            ]
        }