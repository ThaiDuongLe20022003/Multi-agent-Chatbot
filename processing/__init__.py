"""
Processing module for document handling and RAG operations.

This module contains functions for PDF processing, vector database management, 
and Retrieval-Augmented Generation (RAG) chain operations.
"""

from .vector_db import (
    create_simple_vector_db,
    get_simple_retriever, 
    delete_vector_db
)

from .document_processor import (
    extract_all_pages_as_images,
    count_tokens
)

from .rag_chain import (
    process_question_simple,
    generate_response_with_metrics,
    process_question_with_agents,
    simple_llm_call
)

from .multi_agent_chain import (
    process_question_multi_agent
)

__all__ = [
    # Vector database functions
    'create_simple_vector_db',
    'get_simple_retriever', 
    'delete_vector_db',
    
    # Document processing functions
    'extract_all_pages_as_images',
    'count_tokens',
    
    # RAG chain functions
    'process_question_simple',
    'generate_response_with_metrics',
    'process_question_with_agents',
    'simple_llm_call',
    
    # Multi-agent functions
    'process_question_multi_agent'
]