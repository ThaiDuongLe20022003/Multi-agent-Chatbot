"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain with multi-judge evaluation.
Enhanced with advanced similarity search and comprehensive metrics tracking.

This is the main orchestrator that imports and uses all modular components.
"""

import streamlit as st
import ollama
import warnings
import time

# Suppress torch warning
warnings.filterwarnings('ignore', category = UserWarning, message = '.*torch.classes.*')

# Import modular components
from config.settings import STREAMLIT_CONFIG
from data_models.models import ChatMessage
from evaluation.metrics_collector import MetricsCollector
from ui.sidebar import render_sidebar, render_metrics_summary
from ui.chat_interface import render_chat_interface, display_chat_history, handle_user_input
from ui.pdf_viewer import render_pdf_uploader, render_pdf_viewer, render_delete_button, handle_pdf_upload
from utils.helpers import extract_model_names, setup_logging, initialize_session_state
from processing.rag_chain import process_question_simple, process_question_with_agents, count_tokens
from evaluation.evaluator import LLMJudgeEvaluator

def main():
    """Main application function"""
    # Setup application
    setup_logging()
    st.set_page_config(**STREAMLIT_CONFIG)
    initialize_session_state()
    
    # Get available models
    available_models = get_available_models()
    
    # Initialize metrics collector
    if "metrics_collector" not in st.session_state:
        st.session_state["metrics_collector"] = MetricsCollector()
    
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    # Render sidebar
    with st.sidebar:
        selected_model, evaluation_enabled, judge_evaluator, use_multi_agent = render_sidebar(
            available_models, 
            st.session_state.get("selected_model", available_models[0] if available_models else ""),
            st.session_state["metrics_collector"]
        )
        st.session_state["selected_model"] = selected_model
        st.session_state["evaluation_enabled"] = evaluation_enabled
        st.session_state["use_multi_agent"] = use_multi_agent
        st.session_state["judge_evaluator"] = judge_evaluator
        
        # Display metrics summary
        render_metrics_summary(st.session_state["metrics_collector"])
    
    # Main content - PDF upload and viewer
    with col1:
        file_upload = render_pdf_uploader()
        handle_pdf_upload(file_upload)
        render_pdf_viewer()
        render_delete_button()
    
    # Main content - Chat interface
    with col2:
        message_container = st.container(height = 500, border = True)
        
        # Display chat history
        display_chat_history(message_container)
        
        # Chat input and processing
        if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
            handle_user_input_main(prompt, message_container, st.session_state["vector_db"], 
                                 st.session_state["selected_model"], st.session_state["evaluation_enabled"],
                                 st.session_state.get("judge_evaluator"), st.session_state["metrics_collector"],
                                 st.session_state.get("use_multi_agent", False))

def handle_user_input_main(prompt, message_container, vector_db, selected_model, evaluation_enabled, judge_evaluator, metrics_collector, use_multi_agent):
    """Handle user input for main application"""
    try:
        # Add user message to chat
        st.session_state["messages"].append(ChatMessage(role = "user", content = prompt))
        with message_container.chat_message("user", avatar = "😎"):
            st.markdown(prompt)

        # Process and display assistant response
        with message_container.chat_message("assistant", avatar = "🤖"):
            with st.spinner("Processing your question..."):
                if vector_db is not None:
                    start_time = time.time()
                    
                    # Use multi-agent if enabled, otherwise use simple processing
                    # In main.py - update this section
                    if use_multi_agent:
                        context, response, extra_data = process_question_with_agents(
                            prompt, vector_db, selected_model, use_multi_agent = True
                        )
                    else:
                        context, response = process_question_simple(prompt, vector_db, selected_model)  
                        extra_data = {}
                    
                    response_time = time.time() - start_time
                    token_count = count_tokens(response)
                    
                    st.markdown(response)
                    
                    # Prepare message data
                    message_data = {
                        "role": "assistant", 
                        "content": response
                    }
                    
                    # Add multi-agent data if available
                    if "agent_analyses" in extra_data:
                        message_data["agent_analyses"] = extra_data["agent_analyses"]
                        message_data["consensus_score"] = extra_data.get("consensus_score", 0.0)
                    
                    # Record metrics if evaluation is enabled
                    if evaluation_enabled and judge_evaluator:
                        evaluations = judge_evaluator.evaluate_response(prompt, response, context)
                        
                        metrics = metrics_collector.record_metrics(
                            query = prompt,
                            response = response,
                            context = context,
                            response_time = response_time,
                            token_count = token_count,
                            model = selected_model,
                            session_id = "streamlit_session",
                            evaluations = evaluations
                        )
                        
                        # Convert evaluations to dictionaries for session state storage
                        eval_dicts = []
                        for eval_obj in evaluations:
                            eval_dicts.append({
                                "faithfulness": round(eval_obj.faithfulness, 1),
                                "groundedness": round(eval_obj.groundedness, 1),
                                "factual_consistency": round(eval_obj.factual_consistency, 1),
                                "relevance": round(eval_obj.relevance, 1),
                                "completeness": round(eval_obj.completeness, 1),
                                "fluency": round(eval_obj.fluency, 1),
                                "overall_score": round(eval_obj.overall_score, 1),
                                "rating": eval_obj.evaluation_notes.split(":")[0] if ":" in eval_obj.evaluation_notes else "Average",
                                "judge_model": eval_obj.judge_model
                            })
                        
                        message_data["evaluations"] = eval_dicts
                    
                    st.session_state["messages"].append(ChatMessage(**message_data))
                    
                else:
                    st.warning("Please upload a PDF file first.")

    except Exception as e:
        st.error(f"Error: {str(e)}", icon = "⛔️")

def get_available_models():
    """Get available Ollama models"""
    try:
        models_info = ollama.list()
        return extract_model_names(models_info)
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        return tuple()

if __name__ == "__main__":
    main()