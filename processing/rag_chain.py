"""
RAG chain and response generation functions.
"""

import time
import logging
from typing import Tuple, Dict, Any

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from processing.vector_db import get_simple_retriever
from processing.document_processor import count_tokens


logger = logging.getLogger(__name__)


def simple_llm_call(prompt: str, model: str) -> str:
    """Simple LLM call for agent analyses"""
    try:
        llm = ChatOllama(model = model, request_timeout = 120.0)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return f"Analysis unavailable due to error: {str(e)}"


def process_question_simple(question: str, vector_db, selected_model: str) -> Tuple[str, str]:
    """Simple question processing without advanced features"""
    logger.info(f"Simple processing: {question}")
    
    llm = ChatOllama(model = selected_model, request_timeout = 120.0)
    retriever = get_simple_retriever(vector_db)
    
    # Simple prompt template
    template = """You are a professional legal expert. 
    
    CONTEXT INFORMATION:
    {context}
    
    QUESTION: {question}
    
    Please provide a helpful answer based on the context above. If you cannot find the answer in the context, say so.
    
    ANSWER:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    
    # Get context for evaluation
    context_docs = retriever.invoke(question)
    context = "\n\n".join([
        f"Document {i+1}: {doc.page_content[:300]}..."
        for i, doc in enumerate(context_docs[:3])
    ])
    return context, response  


def generate_response_with_metrics(prompt: str, vector_db, selected_model: str, 
                                 evaluation_enabled: bool = False, judge_evaluator = None,
                                 use_multi_agent: bool = False) -> dict:
    """
    Generate response with comprehensive metrics tracking.
    Returns dictionary with response, context, and metrics.
    """
    try:
        if vector_db is not None:
            start_time = time.time()
            
            # Use multi-agent if enabled, otherwise use simple processing
            if use_multi_agent:
                response, context, extra_data = process_question_with_agents(
                    prompt, vector_db, selected_model, use_multi_agent = True
                )
            else:
                response, context = process_question_simple(prompt, vector_db, selected_model)
                extra_data = {}
            
            response_time = time.time() - start_time
            token_count = count_tokens(response)
            
            # Prepare result dictionary
            result = {
                "response": response,
                "context": context,
                "response_time": response_time,
                "token_count": token_count,
                "success": True
            }
            
            # Add multi-agent data if available
            if "agent_analyses" in extra_data:
                result["agent_analyses"] = extra_data["agent_analyses"]
                result["consensus_score"] = extra_data.get("consensus_score", 0.0)
            
            # Add evaluations if enabled
            if evaluation_enabled and judge_evaluator:
                evaluations = judge_evaluator.evaluate_response(prompt, response, context)
                result["evaluations"] = evaluations
            
            return result
        else:
            return {
                "response": "Please upload a PDF file first.",
                "context": "",
                "response_time": 0,
                "token_count": 0,
                "success": False
            }
            
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return {
            "response": f"Error: {str(e)}",
            "context": "",
            "response_time": 0,
            "token_count": 0,
            "success": False
        }


def process_question_with_agents(question: str, vector_db, selected_model: str, 
                               use_multi_agent: bool = False) -> Tuple[str, str, Dict[str, Any]]:
    """Enhanced processing that can use multi-agent system"""
    if use_multi_agent:
        try:
            from processing.multi_agent_chain import process_question_multi_agent
            context, response, extra_data = process_question_multi_agent(question, vector_db, selected_model)  
            return context, response, extra_data  
        except ImportError as e:
            logger.warning(f"Multi-agent system not available, falling back to single agent: {e}")
    
    # Fall back to original single-agent processing
    context, response = process_question_simple(question, vector_db, selected_model)  
    return context, response, {}  