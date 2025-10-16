"""
Quality Assurance Agent with horizontal collaboration capabilities.
Enhanced for multi-agent quality validation and collaborative improvement.
"""

import logging
from typing import Dict, Any, List, Optional
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate

from .base_agent import BaseAgent, AgentContext, AgentResponse


logger = logging.getLogger(__name__)


class QualityAssuranceAgent(BaseAgent):
    """Agent specialized in quality validation, critique, and response improvement with horizontal collaboration"""
    
    def __init__(self, judge_models: List[str] = None):
        super().__init__("quality_assurance")
        self.judge_models = judge_models or []
        self.llms: Dict[str, ChatOllama] = {}  # Multiple LLMs for different judges
        
        # QA validation prompt template
        self.qa_prompt = ChatPromptTemplate.from_template("""
        You are an expert quality assurance specialist for legal AI responses.
        
        ORIGINAL QUERY:
        {query}

        PROPOSED RESPONSE:
        {proposed_response}

        SOURCE CONTEXT:
        {source_context}

        Your task is to critically evaluate and improve this response:

        **VALIDATION CRITERIA:**
        1. **Factual Accuracy**: Are all statements factually correct based on the source context?
        2. **Completeness**: Does it address all aspects of the query?
        3. **Clarity**: Is the response clear, unambiguous, and well-structured?
        4. **Legal Soundness**: Is the legal reasoning sound and properly supported?
        5. **Professionalism**: Is the tone professional and appropriate for legal context?

        **INSTRUCTIONS:**
        - Identify any errors, omissions, or improvements needed
        - Provide specific, actionable feedback
        - Suggest concrete improvements
        - If the response is already high quality, confirm this

        Provide your evaluation in this format:

        **QUALITY ASSESSMENT:**
        [Overall quality rating: Excellent/Good/Fair/Poor]

        **STRENGTHS:**
        - [List key strengths]

        **AREAS FOR IMPROVEMENT:**
        - [List specific issues found]

        **IMPROVED RESPONSE:**
        [Provide the enhanced version of the response with all improvements incorporated]

        **CONFIDENCE NOTES:**
        [Any important caveats or confidence considerations]
        """)
    
    def initialize_judge_models(self, available_models: List[str], selected_model: str):
        """Initialize judge models (all models except the selected one) for unbiased QA"""
        try:
            self.judge_models = [model for model in available_models if model != selected_model]
            
            # Initialize LLMs for each judge model
            for model_name in self.judge_models:
                self.llms[model_name] = ChatOllama(model=model_name, temperature=0.1, request_timeout=120.0)
            
            logger.info(f"Quality assurance agent initialized with {len(self.judge_models)} judge models: {self.judge_models}")
            
        except Exception as e:
            logger.error(f"Failed to initialize quality assurance judge models: {e}")
            raise
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Validate and improve the proposed response using horizontal multi-judge approach"""
        start_time = self._get_timestamp()
        collaborations = []
        
        try:
            if not self.llms:
                return self._create_error_response("Quality assurance agent not initialized with judge models")
                
            query = context.query
            proposed_response = context.metadata.get("proposed_response")
            source_context = context.metadata.get("source_context", "")
            
            if not proposed_response:
                return self._create_error_response("No proposed response available for quality assurance")
            
            # GATHER ADDITIONAL VALIDATION INPUTS through horizontal collaboration
            validation_inputs = self._gather_validation_inputs(query, proposed_response, context, collaborations)
            
            # Perform quality assurance with multiple judges
            qa_results = self._perform_multi_judge_qa(query, proposed_response, source_context, validation_inputs)
            
            # Combine results from multiple judges
            combined_result = self._combine_qa_results(qa_results)
            
            # BROADCAST QA findings to relevant peers
            broadcast_responses = self.broadcast_message(
                "qa_validation_complete",
                {
                    "query": query,
                    "quality_rating": combined_result.get("quality_rating", "unknown"),
                    "improvements_applied": combined_result.get("has_improvements", False),
                    "primary_issues": combined_result.get("primary_issues", []),
                    "judge_consensus": combined_result.get("consensus_level", "medium")
                }
            )
            collaborations.extend(broadcast_responses)
            
            result_data = {
                "original_query": query,
                "original_response": proposed_response,
                "quality_assessments": qa_results,
                "combined_assessment": combined_result,
                "improved_response": combined_result.get("improved_response", proposed_response),
                "has_improvements": combined_result.get("has_improvements", False),
                "quality_rating": combined_result.get("quality_rating", "unknown"),
                "judge_models_used": list(qa_results.keys()),
                "validation_inputs_used": len(validation_inputs),
                "assessment_type": "multi_judge_quality_review"
            }
            
            processing_time = self._calculate_processing_time(start_time)
            logger.info(f"✅ QA completed: {len(qa_results)} judges, {len(collaborations)} collaborations")
            
            return self._create_success_response(
                data=result_data,
                collaborations=collaborations,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Quality assurance error: {e}")
            return self._create_error_response(f"Quality assurance failed: {str(e)}")
    
    def _handle_data_request(self, content: Dict) -> Dict:
        """
        Handle data requests from peer agents
        Example: Summarize Agent requests QA methodology
        """
        try:
            request_type = content.get("request_type", "qa_methodology")
            
            if request_type == "qa_methodology":
                # Provide QA methodology information
                return {
                    "status": "success",
                    "request_type": "qa_methodology",
                    "methodology": {
                        "approach": "multi_judge_validation",
                        "judge_count": len(self.judge_models),
                        "validation_criteria": ["factual_accuracy", "completeness", "clarity", "legal_soundness", "professionalism"],
                        "improvement_strategy": "collaborative_enhancement"
                    },
                    "judge_models": self.judge_models
                }
            
            elif request_type == "quality_metrics":
                # Provide quality metrics data
                qa_data = content.get("qa_data", {})
                metrics = self._calculate_quality_metrics(qa_data)
                
                return {
                    "status": "success",
                    "request_type": "quality_metrics",
                    "quality_metrics": metrics,
                    "overall_quality_score": metrics.get("overall_score", 0),
                    "quality_breakdown": metrics.get("breakdown", {})
                }
            
            else:
                return {
                    "status": "success",
                    "message": "QA data available",
                    "available_data": ["qa_methodology", "quality_metrics", "validation_insights"]
                }
                
        except Exception as e:
            logger.error(f"Error handling data request: {e}")
            return {"status": "error", "message": f"Data request failed: {str(e)}"}
    
    def _handle_analysis_request(self, content: Dict) -> Dict:
        """
        Handle analysis requests from peer agents
        Example: Agent Manager requests system-wide quality analysis
        """
        try:
            request_type = content.get("request_type", "quality_trends")
            quality_data = content.get("quality_data", [])
            
            if request_type == "quality_trends":
                # Analyze quality trends across multiple responses
                trend_analysis = self._analyze_quality_trends(quality_data)
                
                return {
                    "status": "success",
                    "request_type": "quality_trends",
                    "trend_analysis": trend_analysis,
                    "key_insights": trend_analysis.get("insights", []),
                    "recommendations": trend_analysis.get("recommendations", [])
                }
            
            elif request_type == "bottleneck_analysis":
                # Identify quality bottlenecks in the system
                bottleneck_analysis = self._identify_quality_bottlenecks(quality_data)
                
                return {
                    "status": "success",
                    "request_type": "bottleneck_analysis",
                    "bottlenecks_identified": bottleneck_analysis.get("bottlenecks", []),
                    "bottleneck_severity": bottleneck_analysis.get("severity", "low"),
                    "mitigation_suggestions": bottleneck_analysis.get("suggestions", [])
                }
            
            else:
                return {
                    "status": "success",
                    "message": "QA analysis services available",
                    "available_analyses": ["quality_trends", "bottleneck_analysis", "performance_benchmarking"]
                }
                
        except Exception as e:
            logger.error(f"Error handling analysis request: {e}")
            return {"status": "error", "message": f"Analysis request failed: {str(e)}"}
    
    def _handle_validation_request(self, content: Dict) -> Dict:
        """
        Handle validation requests from peer agents
        Example: Legal Agent requests validation of legal reasoning
        """
        try:
            validation_type = content.get("validation_type", "content_quality")
            content_to_validate = content.get("content", "")
            validation_criteria = content.get("criteria", {})
            query_context = content.get("query", "")
            
            if validation_type == "content_quality":
                # Perform comprehensive content quality validation
                validation_result = self._validate_content_quality(content_to_validate, query_context, validation_criteria)
                
                return {
                    "status": "success",
                    "validation_type": "content_quality",
                    "quality_score": validation_result["score"],
                    "is_acceptable": validation_result["is_acceptable"],
                    "major_issues": validation_result["major_issues"],
                    "minor_issues": validation_result["minor_issues"],
                    "improvement_suggestions": validation_result["suggestions"]
                }
            
            elif validation_type == "legal_accuracy":
                # Specialized validation for legal accuracy
                legal_validation = self._validate_legal_accuracy(content_to_validate, validation_criteria)
                
                return {
                    "status": "success",
                    "validation_type": "legal_accuracy",
                    "legal_accuracy_score": legal_validation["score"],
                    "legal_issues": legal_validation["issues"],
                    "legal_improvements": legal_validation["improvements"]
                }
            
            else:
                return {
                    "status": "success",
                    "message": "QA validation services available",
                    "available_validations": ["content_quality", "legal_accuracy", "factual_consistency"]
                }
                
        except Exception as e:
            logger.error(f"Error handling validation request: {e}")
            return {"status": "error", "message": f"Validation request failed: {str(e)}"}
    
    def _handle_clarification(self, content: Dict) -> Dict:
        """
        Handle clarification requests from peer agents
        Example: Summarize Agent requests clarification on QA feedback
        """
        try:
            clarification_type = content.get("clarification_type", "quality_feedback")
            feedback_to_clarify = content.get("feedback", "")
            specific_questions = content.get("questions", [])
            
            if clarification_type == "quality_feedback":
                # Provide clarification on quality feedback
                clarification = self._clarify_quality_feedback(feedback_to_clarify, specific_questions)
                
                return {
                    "status": "success",
                    "clarification_type": "quality_feedback",
                    "clarification_provided": clarification,
                    "questions_addressed": specific_questions if specific_questions else ["general feedback explanation"]
                }
            
            elif clarification_type == "improvement_guidance":
                # Provide detailed improvement guidance
                guidance = self._provide_improvement_guidance(feedback_to_clarify)
                
                return {
                    "status": "success",
                    "clarification_type": "improvement_guidance",
                    "improvement_guidance": guidance,
                    "guidance_level": "detailed" if len(guidance) > 100 else "basic"
                }
            
            else:
                return {
                    "status": "success",
                    "message": "QA clarification services available",
                    "available_clarifications": ["quality_feedback", "improvement_guidance", "validation_criteria"]
                }
                
        except Exception as e:
            logger.error(f"Error handling clarification request: {e}")
            return {"status": "error", "message": f"Clarification failed: {str(e)}"}
    
    def _gather_validation_inputs(self, query: str, proposed_response: str, context: AgentContext, collaborations: List) -> Dict[str, Any]:
        """Gather additional validation inputs through horizontal collaboration"""
        validation_inputs = {}
        
        # Request legal validation from Legal Analyzer Agent
        if "legal_analyzer" in self.peer_agents:
            legal_validation_request = self.send_message(
                "legal_analyzer",
                "validation_request",
                {
                    "validation_type": "legal_accuracy",
                    "content": proposed_response,
                    "criteria": {"query": query, "context": context.metadata.get("source_context", "")}
                }
            )
            if legal_validation_request and "legal_accuracy_score" in legal_validation_request.content:
                validation_inputs["legal_validation"] = legal_validation_request.content
                collaborations.append(legal_validation_request)
        
        # Request retrieval quality analysis from Data Retrieval Agent
        if "data_retriever" in self.peer_agents and context.metadata.get("retrieved_documents"):
            retrieval_analysis_request = self.send_message(
                "data_retriever",
                "analysis_request",
                {
                    "analysis_type": "retrieval_quality",
                    "retrieved_documents": context.metadata.get("retrieved_documents", []),
                    "query": query
                }
            )
            if retrieval_analysis_request and "quality_metrics" in retrieval_analysis_request.content:
                validation_inputs["retrieval_quality"] = retrieval_analysis_request.content
                collaborations.append(retrieval_analysis_request)
        
        return validation_inputs
    
    def _perform_multi_judge_qa(self, query: str, proposed_response: str, source_context: str, validation_inputs: Dict) -> Dict[str, str]:
        """Execute quality assurance using multiple judge models with enhanced context"""
        qa_results = {}
        
        for model_name, llm in self.llms.items():
            try:
                # Enhance prompt with validation inputs
                enhanced_context = self._enhance_validation_context(source_context, validation_inputs)
                
                prompt = self.qa_prompt.format(
                    query=query,
                    proposed_response=proposed_response,
                    source_context=enhanced_context[:2000]  # Limit context length
                )
                
                response = llm.invoke(prompt)
                qa_results[model_name] = response.content.strip()
                logger.info(f"✅ QA completed by judge model: {model_name}")
                
            except Exception as e:
                logger.error(f"QA error with judge model {model_name}: {e}")
                qa_results[model_name] = f"QA assessment failed: {str(e)}"
        
        return qa_results
    
    def _enhance_validation_context(self, source_context: str, validation_inputs: Dict) -> str:
        """Enhance validation context with additional inputs"""
        enhanced_parts = [source_context]
        
        if validation_inputs.get("legal_validation"):
            legal_data = validation_inputs["legal_validation"]
            enhanced_parts.append(f"LEGAL VALIDATION INPUT: Accuracy score: {legal_data.get('legal_accuracy_score', 'N/A')}")
        
        if validation_inputs.get("retrieval_quality"):
            retrieval_data = validation_inputs["retrieval_quality"]
            enhanced_parts.append(f"RETRIEVAL QUALITY INPUT: Overall score: {retrieval_data.get('quality_metrics', {}).get('overall_score', 'N/A')}")
        
        return "\n\n".join(enhanced_parts)
    
    def _combine_qa_results(self, qa_results: Dict[str, str]) -> Dict[str, Any]:
        """Combine results from multiple judges with consensus analysis"""
        if not qa_results:
            return {"has_improvements": False, "improved_response": "", "quality_rating": "unknown"}
        
        improved_responses = []
        quality_ratings = []
        primary_issues = []
        
        for model_name, result in qa_results.items():
            improved_response = self._extract_improved_response(result)
            quality_rating = self._extract_quality_rating(result)
            issues = self._extract_primary_issues(result)
            
            if improved_response and improved_response != result:
                improved_responses.append((model_name, improved_response))
            if quality_rating:
                quality_ratings.append(quality_rating)
            if issues:
                primary_issues.extend(issues)
        
        # Determine consensus
        if improved_responses:
            # Use the most comprehensive improved response
            best_response = max(improved_responses, key=lambda x: len(x[1]))[1]
            return {
                "has_improvements": True,
                "improved_response": best_response,
                "quality_rating": self._calculate_consensus_rating(quality_ratings),
                "primary_issues": list(set(primary_issues))[:3],
                "consensus_level": "strong" if len(improved_responses) > 1 else "moderate",
                "primary_judge": improved_responses[0][0] if improved_responses else list(qa_results.keys())[0]
            }
        else:
            # No improvements suggested
            return {
                "has_improvements": False,
                "improved_response": "",
                "quality_rating": self._calculate_consensus_rating(quality_ratings),
                "primary_issues": list(set(primary_issues))[:3] if primary_issues else ["No major issues identified"],
                "consensus_level": "strong" if len(quality_ratings) > 1 else "moderate",
                "primary_judge": list(qa_results.keys())[0] if qa_results else "none"
            }
    
    def _extract_improved_response(self, qa_result: str) -> str:
        """Extract the improved response from QA result"""
        try:
            # Look for the IMPROVED RESPONSE section
            if "**IMPROVED RESPONSE:**" in qa_result:
                parts = qa_result.split("**IMPROVED RESPONSE:**")
                if len(parts) > 1:
                    improved_section = parts[1]
                    # Extract until next section or end
                    if "**CONFIDENCE NOTES:**" in improved_section:
                        improved_section = improved_section.split("**CONFIDENCE NOTES:**")[0]
                    return improved_section.strip()
            
            # If no improved section found, return empty
            return ""
            
        except Exception as e:
            logger.error(f"Error extracting improved response: {e}")
            return ""
    
    def _extract_quality_rating(self, qa_result: str) -> str:
        """Extract quality rating from QA result"""
        try:
            if "**QUALITY ASSESSMENT:**" in qa_result:
                parts = qa_result.split("**QUALITY ASSESSMENT:**")
                if len(parts) > 1:
                    assessment_line = parts[1].split('\n')[0].strip()
                    for rating in ["Excellent", "Good", "Fair", "Poor"]:
                        if rating in assessment_line:
                            return rating
            return "Unknown"
        except Exception as e:
            logger.error(f"Error extracting quality rating: {e}")
            return "Unknown"
    
    def _extract_primary_issues(self, qa_result: str) -> List[str]:
        """Extract primary issues from QA result"""
        try:
            issues = []
            if "**AREAS FOR IMPROVEMENT:**" in qa_result:
                parts = qa_result.split("**AREAS FOR IMPROVEMENT:**")
                if len(parts) > 1:
                    improvement_section = parts[1]
                    if "**IMPROVED RESPONSE:**" in improvement_section:
                        improvement_section = improvement_section.split("**IMPROVED RESPONSE:**")[0]
                    
                    # Extract bullet points
                    lines = improvement_section.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('-') and len(line) > 2:
                            issues.append(line[1:].strip())
            
            return issues[:3]  # Return top 3 issues
        except Exception as e:
            logger.error(f"Error extracting primary issues: {e}")
            return []
    
    def _calculate_consensus_rating(self, ratings: List[str]) -> str:
        """Calculate consensus quality rating from multiple judges"""
        if not ratings:
            return "Unknown"
        
        rating_scores = {"Excellent": 4, "Good": 3, "Fair": 2, "Poor": 1, "Unknown": 0}
        score_sum = sum(rating_scores.get(rating, 0) for rating in ratings)
        avg_score = score_sum / len(ratings)
        
        if avg_score >= 3.5:
            return "Excellent"
        elif avg_score >= 2.5:
            return "Good"
        elif avg_score >= 1.5:
            return "Fair"
        else:
            return "Poor"
    
    def _calculate_quality_metrics(self, qa_data: Dict) -> Dict[str, Any]:
        """Calculate comprehensive quality metrics"""
        assessments = qa_data.get("quality_assessments", {})
        total_judges = len(assessments)
        
        if total_judges == 0:
            return {"overall_score": 0, "breakdown": {}}
        
        # Calculate scores from ratings
        rating_scores = {"Excellent": 9, "Good": 7, "Fair": 5, "Poor": 3, "Unknown": 5}
        scores = [rating_scores.get(self._extract_quality_rating(assessment), 5) for assessment in assessments.values()]
        avg_score = sum(scores) / len(scores) if scores else 5
        
        return {
            "overall_score": round(avg_score, 1),
            "breakdown": {
                "judge_consensus": "high" if len(set(scores)) <= 2 else "medium",
                "improvement_rate": qa_data.get("has_improvements", False),
                "total_judges": total_judges,
                "score_range": f"{min(scores)}-{max(scores)}" if scores else "N/A"
            }
        }
    
    def _analyze_quality_trends(self, quality_data: List[Dict]) -> Dict[str, Any]:
        """Analyze quality trends across multiple responses"""
        if not quality_data:
            return {"insights": ["Insufficient data for trend analysis"], "recommendations": ["Collect more quality data"]}
        
        scores = [item.get("quality_score", 0) for item in quality_data if "quality_score" in item]
        improvement_rates = [item.get("has_improvements", False) for item in quality_data]
        
        avg_score = sum(scores) / len(scores) if scores else 0
        improvement_rate = sum(improvement_rates) / len(improvement_rates) if improvement_rates else 0
        
        insights = [
            f"Average quality score: {avg_score:.1f}/10",
            f"Improvement applied in {improvement_rate:.1%} of cases",
            f"Quality data collected for {len(quality_data)} responses"
        ]
        
        recommendations = []
        if avg_score < 7:
            recommendations.append("Focus on improving response quality in key areas")
        if improvement_rate < 0.5:
            recommendations.append("Increase application of quality improvements")
        
        return {
            "insights": insights,
            "recommendations": recommendations if recommendations else ["Quality trends are positive"],
            "metrics": {
                "average_score": round(avg_score, 2),
                "improvement_rate": round(improvement_rate, 2),
                "sample_size": len(quality_data)
            }
        }
    
    def _identify_quality_bottlenecks(self, quality_data: List[Dict]) -> Dict[str, Any]:
        """Identify quality bottlenecks in the system"""
        common_issues = []
        for item in quality_data:
            issues = item.get("primary_issues", [])
            common_issues.extend(issues)
        
        from collections import Counter
        issue_counts = Counter(common_issues)
        top_issues = issue_counts.most_common(3)
        
        bottlenecks = [issue for issue, count in top_issues if count > 1]
        
        return {
            "bottlenecks": bottlenecks if bottlenecks else ["No recurring bottlenecks identified"],
            "severity": "high" if any(count > 3 for _, count in top_issues) else "medium" if bottlenecks else "low",
            "suggestions": [
                f"Address recurring issue: {issue}" for issue in bottlenecks
            ] if bottlenecks else ["Continue current quality practices"]
        }
    
    def _validate_content_quality(self, content: str, query: str, criteria: Dict) -> Dict[str, Any]:
        """Validate content quality against specified criteria"""
        # Simplified validation - would be more comprehensive in practice
        content_length = len(content)
        has_structure = any(marker in content for marker in ['**', '-', '1.', '•'])
        addresses_query = any(term in content.lower() for term in query.lower().split()[:3])
        
        score = min(10, (content_length / 100 + (3 if has_structure else 0) + (4 if addresses_query else 0)))
        
        return {
            "score": score,
            "is_acceptable": score >= 6,
            "major_issues": [] if score >= 6 else ["Content may not fully address query"],
            "minor_issues": ["Consider adding more structure"] if not has_structure else [],
            "suggestions": [
                "Increase content depth" if content_length < 200 else "Content length is adequate",
                "Add structural elements" if not has_structure else "Structure is good"
            ]
        }
    
    def _validate_legal_accuracy(self, content: str, criteria: Dict) -> Dict[str, Any]:
        """Validate legal accuracy of content"""
        # Placeholder for legal accuracy validation
        # In practice, this would involve more sophisticated legal checking
        legal_terms = ['contract', 'liability', 'rights', 'obligation', 'breach']
        legal_term_count = sum(1 for term in legal_terms if term in content.lower())
        
        return {
            "score": min(10, legal_term_count * 2),
            "issues": [] if legal_term_count > 0 else ["Limited legal terminology used"],
            "improvements": [
                "Include more specific legal references",
                "Cite relevant legal principles when possible"
            ]
        }
    
    def _clarify_quality_feedback(self, feedback: str, questions: List[str]) -> str:
        """Provide clarification on quality feedback"""
        questions_text = "\n".join([f"- {q}" for q in questions]) if questions else "general quality assessment"
        return f"The quality assessment was based on comprehensive evaluation of multiple criteria including accuracy, completeness, and clarity. Specific feedback addresses: {questions_text}"
    
    def _provide_improvement_guidance(self, feedback: str) -> str:
        """Provide detailed improvement guidance"""
        return "Based on the quality assessment, focus on enhancing clarity through better structure, ensuring all query aspects are addressed, and verifying factual accuracy against source materials. Consider breaking down complex legal concepts into more accessible explanations."
    
    def get_judge_models(self) -> List[str]:
        """Get the list of judge models being used"""
        return self.judge_models
    
    def has_judge_models(self) -> bool:
        """Check if judge models are available"""
        return len(self.llms) > 0
    
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
            "role": "Multi-judge quality assurance and improvement",
            "capabilities": [
                "multi_model_validation",
                "quality_assessment",
                "response_improvement",
                "quality_trend_analysis"
            ],
            "horizontal_features": [
                "collaborative_validation",
                "quality_consensus_calculation",
                "system_bottleneck_identification",
                "peer_quality_services"
            ],
            "validation_criteria": [
                "factual_accuracy",
                "completeness",
                "clarity",
                "legal_soundness",
                "professionalism"
            ],
            "judge_models": self.judge_models
        }