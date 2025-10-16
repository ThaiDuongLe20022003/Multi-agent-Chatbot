"""
Chat interface components for the Streamlit application.
Enhanced with horizontal multi-agent collaboration visualization.
"""

import streamlit as st
import time

from data_models.models import ChatMessage
from processing.rag_chain import generate_response_with_metrics
from evaluation.metrics_collector import MetricsCollector


def render_chat_interface(vector_db, selected_model, evaluation_enabled, judge_evaluator, metrics_collector):
    """Render the main chat interface with horizontal collaboration features"""
    # Display horizontal system header
    st.markdown("### 💬 Chat with Horizontal Multi-Agent System")
    st.info("🤖 Your queries are processed by multiple AI agents working together in parallel!")
    
    message_container = st.container(height=500, border=True)

    # Display chat history with collaboration insights
    display_chat_history(message_container)

    # Chat input and processing
    if prompt := st.chat_input("Ask about your PDF document...", key="chat_input"):
        handle_user_input(prompt, message_container, vector_db, selected_model, 
                         evaluation_enabled, judge_evaluator, metrics_collector)


def display_chat_history(message_container):
    """Display the chat message history with horizontal collaboration insights"""
    # Initialize messages if not exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message.role == "assistant" else "😎"
        
        with message_container.chat_message(message.role, avatar=avatar):
            st.markdown(message.content)
            
            # Show horizontal collaboration insights if available
            display_collaboration_insights(message)
            
            # Show evaluation scores if available
            display_evaluation_scores(message)


def display_collaboration_insights(message):
    """Display horizontal collaboration insights for assistant messages"""
    if message.role == "assistant" and hasattr(message, 'metadata') and message.metadata:
        insights = message.metadata.get('horizontal_insights', {})
        
        if insights:
            # Create an expandable section for collaboration details
            with st.expander("🔍 Collaboration Details", expanded=False):
                
                # Workflow type with icon
                workflow_type = insights.get('workflow_type', 'unknown')
                workflow_icon = get_workflow_icon(workflow_type)
                st.write(f"{workflow_icon} **Workflow**: {workflow_type.replace('_', ' ').title()}")
                
                # Agents involved
                agents_used = insights.get('agents_used', [])
                if agents_used:
                    st.write(f"👥 **Agents Involved**: {', '.join(agents_used)}")
                
                # Performance metrics
                processing_time = insights.get('processing_time', 0)
                response_time = insights.get('response_time', 0)
                collaboration_count = insights.get('collaboration_count', 0)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                with col2:
                    st.metric("Response Time", f"{response_time:.2f}s")
                
                with col3:
                    st.metric("Collaborations", collaboration_count)
                
                # Parallel execution indicator
                if insights.get('parallel_execution', False):
                    st.success("⚡ **Parallel Execution**: Multiple agents worked simultaneously")
                else:
                    st.info("🔄 **Sequential Execution**: Agents worked in sequence")
                
                # Detailed collaboration log
                if collaboration_count > 0:
                    st.write("**Recent Agent Interactions**:")
                    collaboration_log = insights.get('collaboration_log', [])
                    for i, collab in enumerate(collaboration_log[-5:]):  # Show last 5
                        st.write(f"• `{collab.sender}` → `{collab.receiver}`: {collab.message_type}")


def display_evaluation_scores(message):
    """Display evaluation scores for assistant messages"""
    if message.evaluations:
        # Handle both dictionary and object formats
        if isinstance(message.evaluations[0], dict):
            avg_score = sum(eval_obj["overall_score"] for eval_obj in message.evaluations) / len(message.evaluations)
            judge_count = len(message.evaluations)
        else:
            avg_score = sum(eval_obj.overall_score for eval_obj in message.evaluations) / len(message.evaluations)
            judge_count = len(message.evaluations)
        
        # Determine rating color
        if avg_score >= 8.0:
            color = "green"
            emoji = "🎉"
        elif avg_score >= 6.0:
            color = "orange" 
            emoji = "👍"
        else:
            color = "red"
            emoji = "⚠️"
        
        st.caption(f"{emoji} **Quality Score**: {avg_score:.1f}/10.0 ({judge_count} judges)")


def get_workflow_icon(workflow_type):
    """Get appropriate icon for workflow type"""
    icons = {
        "horizontal_parallel": "🔄",
        "horizontal_legal": "⚖️", 
        "horizontal_simple": "⚡",
        "comprehensive_analysis": "🔍",
        "parallel_collaboration": "🤝",
        "legal_collaboration": "📚",
        "simple_retrieval": "🎯"
    }
    return icons.get(workflow_type, "🤖")


def handle_user_input(prompt, message_container, vector_db, selected_model, 
                     evaluation_enabled, judge_evaluator, metrics_collector):
    """Handle user input and generate response with horizontal system"""
    try:
        # Add user message to chat
        st.session_state["messages"].append(ChatMessage(role="user", content=prompt))
        
        with message_container.chat_message("user", avatar="😎"):
            st.markdown(prompt)

        # Check if we should use horizontal system or fallback
        if st.session_state.get("horizontal_system_active", False) and "agent_manager" in st.session_state:
            # Use horizontal multi-agent system
            process_with_horizontal_system(prompt, message_container, selected_model,
                                         evaluation_enabled, judge_evaluator, metrics_collector)
        else:
            # Fallback to traditional RAG system
            process_with_traditional_system(prompt, message_container, vector_db, selected_model,
                                          evaluation_enabled, judge_evaluator, metrics_collector)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}", icon="⛔️")


def process_with_horizontal_system(prompt, message_container, selected_model,
                                 evaluation_enabled, judge_evaluator, metrics_collector):
    """Process query using horizontal multi-agent system"""
    with message_container.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤖 Horizontal agents are collaborating..."):
            start_time = time.time()
            
            agent_manager = st.session_state.agent_manager
            result = agent_manager.execute_workflow(prompt, "streamlit_session")
            
            response_time = time.time() - start_time
            
            if result.success:
                # Display the response
                st.markdown(result.final_response)
                
                # Prepare horizontal insights
                horizontal_insights = {
                    "workflow_type": result.workflow_type,
                    "agents_used": list(result.agent_responses.keys()),
                    "processing_time": result.processing_time,
                    "collaboration_count": len(result.collaboration_log),
                    "parallel_execution": result.parallel_execution,
                    "response_time": response_time,
                    "collaboration_log": result.collaboration_log[-10:]  # Last 10 collaborations
                }
                
                # Record metrics and update session
                update_session_with_metrics(
                    prompt, 
                    result.final_response,
                    response_time,
                    selected_model,
                    metrics_collector,
                    judge_evaluator,
                    horizontal_insights
                )
                
                # Show quick collaboration summary
                show_collaboration_summary(horizontal_insights)
                
            else:
                error_msg = f"❌ Horizontal system error: {', '.join(result.errors)}"
                st.markdown(error_msg)
                st.session_state["messages"].append(
                    ChatMessage(role="assistant", content=error_msg)
                )


def process_with_traditional_system(prompt, message_container, vector_db, selected_model,
                                  evaluation_enabled, judge_evaluator, metrics_collector):
    """Process query using traditional RAG system (fallback)"""
    with message_container.chat_message("assistant", avatar="🤖"):
        with st.spinner("Processing your question..."):
            result = generate_response_with_metrics(
                prompt, vector_db, selected_model, evaluation_enabled, judge_evaluator
            )
            
            st.markdown(result["response"])
            
            # Record metrics and update session state
            if result["success"] and evaluation_enabled and "evaluations" in result:
                update_session_with_metrics(
                    prompt, result, selected_model, metrics_collector
                )
            else:
                # Create ChatMessage object without evaluations
                st.session_state["messages"].append(
                    ChatMessage(role="assistant", content=result["response"])
                )


def show_collaboration_summary(insights):
    """Show a quick summary of agent collaboration"""
    agents_used = insights.get('agents_used', [])
    collaboration_count = insights.get('collaboration_count', 0)
    workflow_type = insights.get('workflow_type', 'unknown')
    
    if agents_used:
        summary_text = f"**Collaboration Summary**: {len(agents_used)} agents worked together"
        if collaboration_count > 0:
            summary_text += f" with {collaboration_count} interactions"
        
        if insights.get('parallel_execution', False):
            summary_text += " using parallel execution"
        
        st.caption(summary_text)


def update_session_with_metrics(prompt, response, response_time, selected_model, metrics_collector, 
                              judge_evaluator, horizontal_insights=None):
    """Update session state with metrics and horizontal insights"""
    try:
        # Get context for evaluation
        if horizontal_insights:
            context = f"Horizontal {horizontal_insights['workflow_type']} workflow with {len(horizontal_insights['agents_used'])} agents"
        else:
            context = "Traditional RAG response"
        
        # Get evaluations from judge if enabled
        evaluations = []
        if judge_evaluator:
            evaluations = judge_evaluator.evaluate_response(prompt, response, context)
        
        # Record metrics
        if evaluations:
            metrics = metrics_collector.record_metrics(
                query=prompt,
                response=response,
                context=context,
                response_time=response_time,
                token_count=len(response.split()),
                model=selected_model,
                session_id="streamlit_session",
                evaluations=evaluations
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
                "rating": eval_obj.evaluation_notes.split(":")[0] if ":" in eval_obj.evaluation_notes else "Unknown",
                "judge_model": eval_obj.judge_model
            })
        
        # Create ChatMessage object with evaluations and horizontal insights
        assistant_message = ChatMessage(
            role="assistant",
            content=response,
            evaluations=eval_dicts
        )
        
        # Add horizontal insights to metadata if available
        if horizontal_insights:
            assistant_message.metadata = {"horizontal_insights": horizontal_insights}
        
        st.session_state["messages"].append(assistant_message)
        
    except Exception as e:
        st.error(f"Error recording metrics: {e}")
        # Fallback: create message without evaluations
        assistant_message = ChatMessage(role="assistant", content=response)
        if horizontal_insights:
            assistant_message.metadata = {"horizontal_insights": horizontal_insights}
        st.session_state["messages"].append(assistant_message)