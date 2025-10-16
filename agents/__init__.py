"""
Multi-agent system for horizontal collaboration.
Enhanced with peer-to-peer communication and parallel execution capabilities.
"""

from .base_agent import BaseAgent, AgentContext, AgentResponse, AgentMessage
from .agent_manager import AgentManager, WorkflowResult
from .pdf_processing_agent import PDFProcessingAgent
from .data_retrieval_agent import DataRetrievalAgent
from .legal_analyzer_agent import LegalAnalyzerAgent
from .summarize_reason_agent import SummarizeReasonAgent
from .quality_assurance_agent import QualityAssuranceAgent

__all__ = [
    # Base classes and data structures
    'BaseAgent',
    'AgentContext', 
    'AgentResponse',
    'AgentMessage',
    
    # Manager and workflow results
    'AgentManager',
    'WorkflowResult',
    
    # Specialized agents with horizontal capabilities
    'PDFProcessingAgent',
    'DataRetrievalAgent',
    'LegalAnalyzerAgent', 
    'SummarizeReasonAgent',
    'QualityAssuranceAgent'
]

# Grouped exports for better organization
BASE_CLASSES = [
    'BaseAgent',
    'AgentContext',
    'AgentResponse', 
    'AgentMessage'
]

MANAGEMENT_CLASSES = [
    'AgentManager',
    'WorkflowResult'
]

SPECIALIZED_AGENTS = [
    'PDFProcessingAgent',
    'DataRetrievalAgent',
    'LegalAnalyzerAgent',
    'SummarizeReasonAgent',
    'QualityAssuranceAgent'
]

# Package metadata for system discovery
PACKAGE_INFO = {
    'name': 'agents',
    'description': 'Horizontal Multi-Agent System with Peer-to-Peer Collaboration',
    'version': '2.0.0',
    'architecture': 'horizontal',
    'features': [
        'parallel_execution',
        'peer_to_peer_communication', 
        'dynamic_workflow_routing',
        'collaborative_decision_making',
        'multi_judge_quality_assurance'
    ],
    'agents_count': 5,
    'horizontal_capabilities': True
}

def get_agent_system_info() -> dict:
    """
    Get comprehensive information about the horizontal multi-agent system.
    Useful for system monitoring and UI display.
    """
    return {
        "system_name": "Horizontal Multi-Agent RAG System",
        "architecture": "Horizontal with Centralized Orchestration",
        "total_agents": len(SPECIALIZED_AGENTS),
        "core_components": {
            "orchestrator": "AgentManager",
            "base_framework": "BaseAgent", 
            "communication_protocol": "AgentMessage",
            "workflow_tracking": "WorkflowResult"
        },
        "agent_roles": {
            "PDFProcessingAgent": "Document extraction and chunking",
            "DataRetrievalAgent": "Semantic search and retrieval",
            "LegalAnalyzerAgent": "Legal analysis and reasoning",
            "SummarizeReasonAgent": "Multi-source synthesis", 
            "QualityAssuranceAgent": "Multi-judge quality validation"
        },
        "horizontal_features": {
            "parallel_execution": "Multiple agents run concurrently",
            "peer_communication": "Direct agent-to-agent messaging",
            "dynamic_routing": "Intelligent workflow selection",
            "collaborative_qa": "Multi-agent quality assurance",
            "performance_tracking": "Processing time and collaboration metrics"
        },
        "workflow_types": [
            "parallel_collaboration",
            "legal_collaboration", 
            "simple_retrieval",
            "comprehensive_analysis"
        ]
    }

def list_agent_capabilities() -> dict:
    """
    List capabilities of all agents in the horizontal system.
    Useful for dynamic agent discovery and capability-based routing.
    """
    return {
        "PDFProcessingAgent": {
            "primary_role": "PDF text extraction and document preparation",
            "capabilities": [
                "pdf_text_extraction",
                "document_chunking",
                "metadata_enrichment", 
                "structure_analysis",
                "peer_collaboration"
            ],
            "horizontal_features": [
                "data_request_handling",
                "analysis_request_handling",
                "collaboration_request_handling", 
                "broadcast_messaging"
            ]
        },
        "DataRetrievalAgent": {
            "primary_role": "Semantic search and document retrieval",
            "capabilities": [
                "semantic_search",
                "vector_database_management",
                "specialized_retrieval", 
                "retrieval_quality_analysis"
            ],
            "horizontal_features": [
                "specialized_data_requests",
                "retrieval_analysis", 
                "validation_services",
                "broadcast_messaging"
            ],
            "retrieval_strategies": [
                "semantic_similarity",
                "legal_enhanced_search",
                "comprehensive_search"
            ]
        },
        "LegalAnalyzerAgent": {
            "primary_role": "Legal analysis and reasoning specialist",
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
        },
        "SummarizeReasonAgent": {
            "primary_role": "Information synthesis and reasoning specialist",
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
        },
        "QualityAssuranceAgent": {
            "primary_role": "Multi-judge quality assurance and improvement",
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
            ]
        }
    }