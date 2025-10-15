"""
Chat interface components for the Streamlit application.
"""

import streamlit as st

from data_models.models import ChatMessage
from processing.rag_chain import generate_response_with_metrics
from evaluation.metrics_collector import MetricsCollector


def render_chat_interface(vector_db, selected_model, evaluation_enabled, judge_evaluator, metrics_collector):
    """Render the main chat interface"""
    message_container = st.container(height = 500, border = True)

    # Display chat history
    display_chat_history(message_container)

    # Chat input and processing
    if prompt := st.chat_input("Enter a prompt here...", key = "chat_input"):
        handle_user_input(prompt, message_container, vector_db, selected_model, 
                         evaluation_enabled, judge_evaluator, metrics_collector)


def display_chat_history(message_container):
    """Display the chat message history"""
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message.role == "assistant" else "😎"  # Use dot notation
        with message_container.chat_message(message.role, avatar = avatar):  # Use dot notation
            st.markdown(message.content)  # Use dot notation
            
            # Show evaluation scores if available
            if message.evaluations:  # Use dot notation
                # Handle both dictionary and object formats
                if isinstance(message.evaluations[0], dict):  # Use dot notation
                    avg_score = sum(eval_obj["overall_score"] for eval_obj in message.evaluations) / len(message.evaluations)
                else:
                    avg_score = sum(eval_obj.overall_score for eval_obj in message.evaluations) / len(message.evaluations)
                st.caption(f"📊 Average Evaluation: {avg_score:.1f}/10.0 ({len(message.evaluations)} judges)")


def handle_user_input(prompt, message_container, vector_db, selected_model, 
                     evaluation_enabled, judge_evaluator, metrics_collector):
    """Handle user input and generate response"""
    try:
        # Add user message to chat - create ChatMessage object
        st.session_state["messages"].append(ChatMessage(role = "user", content = prompt))
        with message_container.chat_message("user", avatar = "😎"):
            st.markdown(prompt)

        # Process and display assistant response
        with message_container.chat_message("assistant", avatar = "🤖"):
            with st.spinner("Processing your question..."):
                result = generate_response_with_metrics(
                    prompt, vector_db, selected_model, evaluation_enabled, judge_evaluator
                )
                
                st.markdown(result["response"])
                
                # Record metrics and update session state
                if result["success"] and evaluation_enabled and "evaluations" in result:
                    update_session_with_metrics(prompt, result, selected_model, metrics_collector)
                else:
                    # Create ChatMessage object without evaluations
                    st.session_state["messages"].append(
                        ChatMessage(role = "assistant", content=result["response"])
                    )

    except Exception as e:
        st.error(f"Error: {str(e)}", icon = "⛔️")


def update_session_with_metrics(prompt, result, selected_model, metrics_collector):
    """Update session state with metrics and evaluations"""
    # Record metrics
    metrics = metrics_collector.record_metrics(
        query = prompt,
        response = result["response"],
        context = result["context"],
        response_time = result["response_time"],
        token_count = result["token_count"],
        model = selected_model,
        session_id = "streamlit_session",
        evaluations = result["evaluations"]
    )
    
    # Convert evaluations to dictionaries for session state storage
    eval_dicts = []
    for eval_obj in result["evaluations"]:
        eval_dicts.append({
            "faithfulness": round(eval_obj.faithfulness, 1),
            "groundedness": round(eval_obj.groundedness, 1),
            "factual_consistency": round(eval_obj.factual_consistency, 1),
            "relevance": round(eval_obj.relevance, 1),
            "completeness": round(eval_obj.completeness, 1),
            "fluency": round(eval_obj.fluency, 1),
            "overall_score": round(eval_obj.overall_score, 1),
            "rating": eval_obj.evaluation_notes.split(":")[0],
            "judge_model": eval_obj.judge_model
        })
    
    # Create ChatMessage object with evaluations
    st.session_state["messages"].append(
        ChatMessage(role = "assistant", content = result["response"], evaluations = eval_dicts)
    )