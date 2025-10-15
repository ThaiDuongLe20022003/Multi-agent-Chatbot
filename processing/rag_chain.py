"""
RAG chain and response generation functions.
"""

import time
import logging
from typing import Tuple

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from processing.vector_db import get_simple_retriever
from processing.document_processor import count_tokens


logger = logging.getLogger(__name__)


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
    
    return response, context


def generate_response_with_metrics(prompt: str, vector_db, selected_model: str, 
                                 evaluation_enabled: bool, judge_evaluator=None) -> dict:
    """
    Generate response with comprehensive metrics tracking.
    Returns dictionary with response, context, and metrics.
    """
    try:
        if vector_db is not None:
            start_time = time.time()
            response, context = process_question_simple(prompt, vector_db, selected_model)
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