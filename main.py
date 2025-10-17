"""
Streamlit application for PDF-based RAG using Horizontal Multi-Agent System.
Enhanced with peer-to-peer collaboration and parallel execution.
"""

import streamlit as st
import ollama
import warnings
import time
from typing import List

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# Import modular components
from config import STREAMLIT_CONFIG
from data_models import ChatMessage
from evaluation import MetricsCollector
from ui import render_sidebar, render_metrics_summary
from ui import render_pdf_uploader, render_pdf_viewer, render_delete_button
from utils import setup_logging, initialize_session_state, extract_model_names

# Import horizontal multi-agent system
from agents import (
    AgentManager,
    PDFProcessingAgent,
    DataRetrievalAgent,
    LegalAnalyzerAgent,
    SummarizeReasonAgent,
    QualityAssuranceAgent,
    get_agent_system_info,
    list_agent_capabilities
)


def main():
    """Main application function with horizontal multi-agent integration"""
    # Setup application
    setup_logging()
    st.set_page_config(**STREAMLIT_CONFIG)
    initialize_session_state()
    
    # Get available models
    available_models = get_available_models()
    
    # Initialize horizontal multi-agent system
    initialize_horizontal_agent_system(available_models)
    
    # Initialize metrics collector
    if "metrics_collector" not in st.session_state:
        st.session_state["metrics_collector"] = MetricsCollector()
    
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    # Render sidebar with horizontal system info
    with st.sidebar:
        selected_model, evaluation_enabled, judge_evaluator = render_sidebar(
            available_models,
            st.session_state.get("selected_model", available_models[0] if available_models else ""),
            st.session_state["metrics_collector"]
        )
        
        # Update selected model in session state
        st.session_state["selected_model"] = selected_model
        st.session_state["evaluation_enabled"] = evaluation_enabled
        
        # Update agent models if selection changed
        update_agent_models(selected_model, available_models)
        
        # Display horizontal agent system status
        render_horizontal_system_status()
        
        # Display metrics summary
        render_metrics_summary(st.session_state["metrics_collector"])
    
    # Main content - PDF upload and viewer
    with col1:
        file_upload = render_pdf_uploader()
        handle_pdf_upload_horizontal(file_upload)
        render_pdf_viewer()
        render_delete_button_horizontal()
    
    # Main content - Chat interface with horizontal collaboration
    with col2:
        render_chat_interface_horizontal(
            st.session_state["selected_model"],
            st.session_state["evaluation_enabled"],
            judge_evaluator,
            st.session_state["metrics_collector"]
        )


def initialize_horizontal_agent_system(available_models: List[str]):
    """Initialize the horizontal multi-agent system with all agents"""
    if "agent_manager" not in st.session_state:
        # Create horizontal agent manager
        st.session_state.agent_manager = AgentManager()
        
        # Register all horizontal agents
        pdf_agent = PDFProcessingAgent()
        data_agent = DataRetrievalAgent()
        legal_agent = LegalAnalyzerAgent()
        summarize_agent = SummarizeReasonAgent()
        
        # Initialize QA agent with judge models
        qa_agent = QualityAssuranceAgent(judge_models=[])
        
        # Register agents with horizontal peer connections
        st.session_state.agent_manager.register_agent(pdf_agent)
        st.session_state.agent_manager.register_agent(data_agent)
        st.session_state.agent_manager.register_agent(legal_agent)
        st.session_state.agent_manager.register_agent(summarize_agent)
        st.session_state.agent_manager.register_agent(qa_agent)
        
        # If no Ollama models are available, inform the user
        if not available_models:
            st.error(
                "No Ollama models detected. Please install an Ollama model locally and restart the app.\n\n"
                "Example (run in your terminal): `ollama pull llama2`\n\n"
                "After installing a model, refresh this app to use the agents with the installed model."
            )
            st.session_state.agent_system_initialized = False
            print("⚠️ Horizontal agent system initialized without models")
            return

        # Initialize with the first available model
        default_model = available_models[0]
        update_agent_models(default_model, available_models)
        
        st.session_state.agent_system_initialized = True
        st.session_state.horizontal_system_active = True
        
        print(f"🚀 Horizontal multi-agent system initialized with 5 agents using model: {default_model}")
        print(f"📊 System info: {get_agent_system_info()}")


def update_agent_models(selected_model: str, available_models: List[str]):
    """Update all agents with the selected model and available models"""
    if "agent_manager" not in st.session_state:
        return
        
    agent_manager = st.session_state.agent_manager
    
    # Update legal analyzer agent
    legal_agent = agent_manager.get_agent("legal_analyzer")
    if legal_agent:
        try:
            legal_agent.initialize_model(selected_model)
            print(f"✅ Legal analyzer agent updated with model: {selected_model}")
        except Exception as e:
            st.error(f"Failed to initialize legal analyzer with {selected_model}: {e}")
    
    # Update summarize reason agent
    summarize_agent = agent_manager.get_agent("summarize_reason")
    if summarize_agent:
        try:
            summarize_agent.initialize_model(selected_model)
            print(f"✅ Summarize reason agent updated with model: {selected_model}")
        except Exception as e:
            st.error(f"Failed to initialize summarize reason agent with {selected_model}: {e}")
    
    # Update quality assurance agent with judge models
    qa_agent = agent_manager.get_agent("quality_assurance")
    if qa_agent and available_models:
        try:
            # QA agent uses all models EXCEPT the selected one for unbiased evaluation
            qa_agent.initialize_judge_models(available_models, selected_model)
            judge_models = qa_agent.get_judge_models()
            print(f"✅ QA Agent using {len(judge_models)} judge models: {judge_models}")
        except Exception as e:
            st.error(f"Failed to initialize quality assurance agent with judge models: {e}")


def render_horizontal_system_status():
    """Display horizontal multi-agent system status in sidebar"""
    st.header("🤖 Horizontal Multi-Agent System")
    
    if "agent_manager" in st.session_state and st.session_state.get("horizontal_system_active", False):
        agent_manager = st.session_state.agent_manager
        
        # Display system overview
        system_info = get_agent_system_info()
        st.subheader("🏗️ System Architecture")
        st.write(f"**Architecture**: {system_info['architecture']}")
        st.write(f"**Total Agents**: {system_info['total_agents']}")
        st.write(f"**Active Workflows**: {len(system_info['workflow_types'])}")
        
        # Health check for all agents
        st.subheader("🟢 Agent Status")
        health_status = agent_manager.health_check_all()
        for agent_name, is_healthy in health_status.items():
            status_icon = "✅" if is_healthy else "❌"
            status_color = "green" if is_healthy else "red"
            st.write(f"{status_icon} **{agent_name}**")
        
        # Workflow statistics
        stats = agent_manager.get_workflow_stats()
        st.subheader("📊 Workflow Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Executions", stats["total_executions"])
            st.metric("Success Rate", f"{stats['success_rate']:.1%}")
            st.metric("Parallel Executions", stats["parallel_executions"])
        
        with col2:
            st.metric("Avg Processing Time", f"{stats['average_processing_time']}s")
            st.metric("Total Collaborations", stats["total_collaborations"])
            st.metric("Cache Size", stats.get("cache_size", 0))
        
        # Agent usage breakdown
        if stats["agent_usage"]:
            st.subheader("👥 Agent Usage")
            for agent, count in stats["agent_usage"].items():
                usage_percent = (count / stats["total_executions"]) * 100 if stats["total_executions"] > 0 else 0
                st.write(f"• **{agent}**: {count} uses ({usage_percent:.1f}%)")
        
        # Recent workflow types
        if stats.get("workflow_types"):
            st.subheader("🔄 Recent Workflows")
            for workflow in stats["workflow_types"][-3:]:  # Last 3 workflows
                st.write(f"• {workflow.replace('_', ' ').title()}")
        
        # System capabilities
        with st.expander("🔧 System Capabilities"):
            capabilities = list_agent_capabilities()
            for agent_name, agent_caps in capabilities.items():
                st.write(f"**{agent_name}**")
                st.write(f"Role: {agent_caps['primary_role']}")
                st.write(f"Features: {', '.join(agent_caps['horizontal_features'])}")
                st.write("---")
                
    else:
        st.warning("⚠️ Horizontal system not active")
        if st.button("🔄 Initialize Horizontal System"):
            st.rerun()


def handle_pdf_upload_horizontal(file_upload):
    """Handle PDF upload using horizontal agent system"""
    if file_upload and st.session_state.get("vector_db") is None:
        with st.spinner("🤖 Horizontal agents are processing your PDF..."):
            try:
                # Use PDF Processing Agent from horizontal system
                agent_manager = st.session_state.agent_manager
                pdf_agent = agent_manager.get_agent("pdf_processor")
                
                if not pdf_agent:
                    st.error("PDF Processing Agent not available in horizontal system")
                    return
                
                # Create context for PDF processing
                from agents.base_agent import AgentContext
                context = AgentContext(
                    query = "process_pdf",
                    session_id = "pdf_upload",
                    metadata = {"pdf_file": file_upload, "has_pdf": True}
                )
                
                # Process PDF with horizontal agent
                result = pdf_agent.process(context)
                
                if result.success:
                    chunks = result.data["chunks"]
                    
                    # Initialize vector DB with Data Retrieval Agent
                    data_agent = agent_manager.get_agent("data_retriever")
                    if data_agent and data_agent.initialize_vector_db(chunks, f"pdf_{hash(file_upload.name)}"):
                        st.session_state["vector_db"] = data_agent.vector_db
                        st.session_state["file_upload"] = file_upload
                        
                        # Extract pages for viewer using existing function
                        from processing.document_processor import extract_all_pages_as_images
                        with st.session_state["file_upload"] as pdf_file:
                            st.session_state["pdf_pages"] = extract_all_pages_as_images(pdf_file)
                        
                        # Show collaboration insights
                        collaboration_count = len(result.collaborations)
                        if collaboration_count > 0:
                            st.success(f"✅ PDF processed successfully by {collaboration_count} collaborating agents!")
                        else:
                            st.success("✅ PDF processed successfully by AI agents!")
                    else:
                        st.error("❌ Failed to initialize vector database in horizontal system")
                else:
                    st.error(f"❌ PDF processing failed: {result.error_message}")
                    
            except Exception as e:
                st.error(f"❌ Error processing PDF in horizontal system: {str(e)}")


def render_chat_interface_horizontal(selected_model, evaluation_enabled, judge_evaluator, metrics_collector):
    """Chat interface using horizontal multi-agent system"""
    message_container = st.container(height = 500, border = True)

    # Display chat history
    display_chat_history_horizontal(message_container)

    # Chat input and processing
    if prompt := st.chat_input("Ask about your PDF document...", key = "chat_input_horizontal"):
        handle_user_input_horizontal(prompt, message_container, selected_model,
                                   evaluation_enabled, judge_evaluator, metrics_collector)


def display_chat_history_horizontal(message_container):
    """Display the chat message history with horizontal collaboration insights"""
    # Initialize messages if not exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message.role == "assistant" else "😎"
        with message_container.chat_message(message.role, avatar=avatar):
            st.markdown(message.content)
            
            # Show horizontal collaboration insights if available
            if hasattr(message, 'metadata') and message.metadata.get('horizontal_insights'):
                insights = message.metadata['horizontal_insights']
                with st.expander("🤝 Collaboration Insights"):
                    st.write(f"**Workflow**: {insights.get('workflow_type', 'Unknown')}")
                    st.write(f"**Agents Used**: {', '.join(insights.get('agents_used', []))}")
                    st.write(f"**Processing Time**: {insights.get('processing_time', 0):.2f}s")
                    
                    if insights.get('collaboration_count', 0) > 0:
                        st.write(f"**Collaborations**: {insights['collaboration_count']} interactions")
            
            # Show evaluation scores if available
            if message.evaluations:
                if isinstance(message.evaluations[0], dict):
                    avg_score = sum(eval_obj["overall_score"] for eval_obj in message.evaluations) / len(message.evaluations)
                else:
                    avg_score = sum(eval_obj.overall_score for eval_obj in message.evaluations) / len(message.evaluations)
                st.caption(f"📊 Average Evaluation: {avg_score:.1f}/10.0 ({len(message.evaluations)} judges)")


def handle_user_input_horizontal(prompt, message_container, selected_model,
                               evaluation_enabled, judge_evaluator, metrics_collector):
    """Handle user input using horizontal multi-agent system"""
    try:
        # Add user message to chat
        st.session_state["messages"].append(ChatMessage(role = "user", content = prompt))
        
        with message_container.chat_message("user", avatar = "😎"):
            st.markdown(prompt)

        # Process with horizontal multi-agent system
        with message_container.chat_message("assistant", avatar = "🤖"):
            with st.spinner("🤖 Horizontal agents are collaborating..."):
                start_time = time.time()
                
                agent_manager = st.session_state.agent_manager
                result = agent_manager.execute_workflow(prompt, "streamlit_session")
                
                response_time = time.time() - start_time
                
                if result.success:
                    # Display the response with horizontal insights
                    st.markdown(result.final_response)
                    
                    # Create ChatMessage with horizontal metadata
                    horizontal_insights = {
                        "workflow_type": result.workflow_type,
                        "agents_used": list(result.agent_responses.keys()),
                        "processing_time": result.processing_time,
                        "collaboration_count": len(result.collaboration_log),
                        "parallel_execution": result.parallel_execution,
                        "response_time": response_time
                    }
                    
                    # Record metrics if evaluation enabled
                    if evaluation_enabled and judge_evaluator:
                        update_session_with_metrics_horizontal(
                            prompt,
                            result.final_response,
                            response_time,
                            selected_model,
                            metrics_collector,
                            judge_evaluator,
                            horizontal_insights
                        )
                    else:
                        # Create ChatMessage with horizontal insights
                        assistant_message = ChatMessage(
                            role = "assistant",
                            content=result.final_response
                        )
                        # Add horizontal insights to metadata
                        assistant_message.metadata = {"horizontal_insights": horizontal_insights}
                        st.session_state["messages"].append(assistant_message)
                    
                    # Show horizontal collaboration details
                    with st.expander("🔍 View Agent Collaboration Details"):
                        st.write(f"**Workflow Type**: {result.workflow_type}")
                        st.write(f"**Agents Involved**: {', '.join(result.agent_responses.keys())}")
                        st.write(f"**Total Processing Time**: {result.processing_time:.2f}s")
                        st.write(f"**Collaboration Interactions**: {len(result.collaboration_log)}")
                        
                        if result.collaboration_log:
                            st.write("**Recent Collaborations**:")
                            for i, collab in enumerate(result.collaboration_log[-3:]):  # Show last 3
                                st.write(f"- {collab.sender} → {collab.receiver}: {collab.message_type}")
                        
                        if result.parallel_execution:
                            st.success("⚡ Parallel execution used")
                        else:
                            st.info("🔄 Sequential execution used")
                        
                else:
                    error_msg = f"❌ Horizontal system error: {', '.join(result.errors)}"
                    st.markdown(error_msg)
                    st.session_state["messages"].append(
                        ChatMessage(role = "assistant", content = error_msg)
                    )

    except Exception as e:
        st.error(f"❌ Error in horizontal system: {str(e)}", icon = "⛔️")


def update_session_with_metrics_horizontal(prompt, response, response_time, selected_model,
                                         metrics_collector, judge_evaluator, horizontal_insights):
    """Update session state with metrics and horizontal insights"""
    try:
        # Get context for evaluation
        context = f"Horizontal multi-agent response using {horizontal_insights['workflow_type']} workflow"
        
        # Get evaluations from judge
        evaluations = judge_evaluator.evaluate_response(prompt, response, context)
        
        # Record metrics
        metrics = metrics_collector.record_metrics(
            query = prompt,
            response = response,
            context = context,
            response_time = response_time,
            token_count = len(response.split()),
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
                "rating": eval_obj.evaluation_notes.split(":")[0] if ":" in eval_obj.evaluation_notes else "Unknown",
                "judge_model": eval_obj.judge_model
            })
        
        # Create ChatMessage object with evaluations and horizontal insights
        assistant_message = ChatMessage(
            role = "assistant",
            content = response,
            evaluations = eval_dicts
        )
        # Add horizontal insights to metadata
        assistant_message.metadata = {"horizontal_insights": horizontal_insights}
        
        st.session_state["messages"].append(assistant_message)
        
    except Exception as e:
        st.error(f"Error recording metrics: {e}")
        # Fallback: create message without evaluations
        assistant_message = ChatMessage(role = "assistant", content = response)
        assistant_message.metadata = {"horizontal_insights": horizontal_insights}
        st.session_state["messages"].append(assistant_message)


def render_delete_button_horizontal():
    """Delete button using horizontal agent system"""
    delete_collection = st.button(
        "⚠️ Delete collection",
        type = "secondary",
        key = "delete_button_horizontal"
    )

    if delete_collection:
        from processing.vector_db import delete_vector_db
        delete_vector_db(st.session_state.get("vector_db"))
        
        # Also clear agent vector DB
        if "agent_manager" in st.session_state:
            data_agent = st.session_state.agent_manager.get_agent("data_retriever")
            if data_agent:
                data_agent.clear_vector_db()
        
        # Clear horizontal system cache if exists
        if "agent_manager" in st.session_state:
            cache_size = st.session_state.agent_manager.clear_cache()
            if cache_size > 0:
                st.info(f"🧹 Cleared {cache_size} cached responses")
        
        st.rerun()


def get_available_models():
    """Get available Ollama models with comprehensive error handling"""
    try:
        models_info = ollama.list()
        models = extract_model_names(models_info)
        
        if not models:
            st.warning("⚠️ Ollama is running but no models are installed.")
            return []
            
        print(f"🤖 Found {len(models)} models: {', '.join(models)}")
        return models
        
    except ollama.ResponseError as e:
        if 'connection refused' in str(e).lower():
            st.error("🔌 Ollama is not running. Please start Ollama first.")
            st.code("ollama serve", language="bash")
        else:
            st.error(f"❌ Ollama error: {e}")
        return []
    except Exception as e:
        st.error(f"❌ Unexpected error connecting to Ollama: {e}")
        st.write("Please ensure Ollama is installed and running.")
        return []


if __name__ == "__main__":
    main()