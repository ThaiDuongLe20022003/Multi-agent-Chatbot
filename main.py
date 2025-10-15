"""
Streamlit application for PDF-based RAG using Multi-Agent System.
Enhanced with horizontal multi-agent architecture.
"""

import streamlit as st
import ollama
import warnings

# Suppress torch warning
warnings.filterwarnings('ignore', category = UserWarning, message = '.*torch.classes.*')

# Import modular components
from config import STREAMLIT_CONFIG
from data_models import ChatMessage
from evaluation import MetricsCollector
from ui import render_sidebar, render_metrics_summary, render_chat_interface
from ui import render_pdf_uploader, render_pdf_viewer, render_delete_button, handle_pdf_upload
from utils import setup_logging, initialize_session_state, extract_model_names

# Import multi-agent system
from agents.agent_manager import AgentManager
from agents.pdf_processing_agent import PDFProcessingAgent
from agents.data_retrieval_agent import DataRetrievalAgent


def main():
    """Main application function"""
    # Setup application
    setup_logging()
    st.set_page_config(**STREAMLIT_CONFIG)
    initialize_session_state()  
    
    # Initialize multi-agent system
    initialize_agent_system()
    
    # Get available models
    available_models = get_available_models()
    
    # Initialize metrics collector
    if "metrics_collector" not in st.session_state:
        st.session_state["metrics_collector"] = MetricsCollector()
    
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    # Render sidebar
    with st.sidebar:
        selected_model, evaluation_enabled, judge_evaluator = render_sidebar(
            available_models, 
            st.session_state.get("selected_model", available_models[0] if available_models else ""),
            st.session_state["metrics_collector"]
        )
        st.session_state["selected_model"] = selected_model
        st.session_state["evaluation_enabled"] = evaluation_enabled
        
        # Display agent system status
        render_agent_system_status()
        
        # Display metrics summary
        render_metrics_summary(st.session_state["metrics_collector"])
    
    # Main content - PDF upload and viewer
    with col1:
        file_upload = render_pdf_uploader()
        handle_pdf_upload_agent(file_upload)  # Updated to use agent system
        render_pdf_viewer()
        render_delete_button_agent()  # Updated to use agent system
    
    # Main content - Chat interface
    with col2:
        render_chat_interface_agent(  # Updated to use agent system
            st.session_state["selected_model"],
            st.session_state["evaluation_enabled"],
            judge_evaluator,
            st.session_state["metrics_collector"]
        )


def initialize_agent_system():
    """Initialize the multi-agent system"""
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = AgentManager()
        
        # Register agents
        pdf_agent = PDFProcessingAgent()
        data_agent = DataRetrievalAgent()
        
        st.session_state.agent_manager.register_agent(pdf_agent)
        st.session_state.agent_manager.register_agent(data_agent)
        
        st.session_state.agent_system_initialized = True
        print("🤖 Multi-agent system initialized with 2 agents")


def handle_pdf_upload_agent(file_upload):
    """Handle PDF upload using agent system"""
    if file_upload and st.session_state["vector_db"] is None:
        with st.spinner("🤖 Agent is processing your PDF..."):
            try:
                # Use PDF Processing Agent
                agent_manager = st.session_state.agent_manager
                pdf_agent = agent_manager.get_agent("pdf_processor")
                
                # Create context for PDF processing
                from agents.base_agent import AgentContext
                context = AgentContext(
                    query = "process_pdf", 
                    session_id = "pdf_upload",
                    metadata = {"pdf_file": file_upload, "has_pdf": True}
                )
                
                # Process PDF with agent
                result = pdf_agent.process(context)
                
                if result.success:
                    chunks = result.data["chunks"]
                    
                    # Initialize vector DB with Data Retrieval Agent
                    data_agent = agent_manager.get_agent("data_retriever")
                    if data_agent.initialize_vector_db(chunks, f"pdf_{hash(file_upload.name)}"):
                        st.session_state["vector_db"] = data_agent.vector_db
                        st.session_state["file_upload"] = file_upload
                        
                        # Extract pages for viewer (using existing function)
                        from processing.document_processor import extract_all_pages_as_images
                        with st.session_state["file_upload"] as pdf_file:
                            st.session_state["pdf_pages"] = extract_all_pages_as_images(pdf_file)
                        
                        st.success("✅ PDF processed successfully by AI agents!")
                    else:
                        st.error("❌ Failed to initialize vector database")
                else:
                    st.error(f"❌ PDF processing failed: {result.error_message}")
                    
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")


def render_agent_system_status():
    """Display agent system status in sidebar"""
    st.header("🤖 Agent System")
    
    if "agent_manager" in st.session_state:
        agent_manager = st.session_state.agent_manager
        
        # Health check
        health_status = agent_manager.health_check_all()
        
        st.subheader("Agent Status")
        for agent_name, is_healthy in health_status.items():
            status_icon = "✅" if is_healthy else "❌"
            st.write(f"{status_icon} {agent_name}")
        
        # Workflow stats
        stats = agent_manager.get_workflow_stats()
        st.subheader("System Stats")
        st.write(f"Total Executions: {stats['total_executions']}")
        st.write(f"Success Rate: {stats['success_rate']:.1%}")
        
        # Agent info
        st.subheader("Available Agents")
        st.write("• PDF Processing Agent")
        st.write("• Data Retrieval Agent")
        st.info("Phase 1: Core Foundation Active")
    else:
        st.warning("Agent system not initialized")


def render_chat_interface_agent(selected_model, evaluation_enabled, judge_evaluator, metrics_collector):
    """Chat interface using agent system"""
    message_container = st.container(height = 500, border = True)

    # Display chat history
    from ui.chat_interface import display_chat_history
    display_chat_history(message_container)

    # Chat input and processing
    if prompt := st.chat_input("Ask about your PDF document...", key = "chat_input"):
        handle_user_input_agent(prompt, message_container, selected_model, 
                              evaluation_enabled, judge_evaluator, metrics_collector)


def handle_user_input_agent(prompt, message_container, selected_model, 
                           evaluation_enabled, judge_evaluator, metrics_collector):
    """Handle user input using agent system"""
    try:
        # Add user message to chat
        from data_models.models import ChatMessage
        st.session_state["messages"].append(ChatMessage(role = "user", content = prompt))
        
        with message_container.chat_message("user", avatar = "😎"):
            st.markdown(prompt)

        # Process with agent system
        with message_container.chat_message("assistant", avatar = "🤖"):
            with st.spinner("🤖 Agents are collaborating..."):
                agent_manager = st.session_state.agent_manager
                result = agent_manager.execute_workflow(prompt, "streamlit_session")
                
                if result.success:
                    # For now, display retrieved documents
                    final_response = result.final_response
                    
                    if isinstance(final_response, dict) and "retrieved_documents" in final_response:
                        # Format the response from retrieved documents
                        docs = final_response["retrieved_documents"]
                        response_text = self._format_agent_response(docs, prompt)
                    else:
                        response_text = str(final_response)
                    
                    st.markdown(response_text)
                    
                    # Record metrics if evaluation enabled
                    if evaluation_enabled and judge_evaluator:
                        from ui.chat_interface import update_session_with_metrics
                        update_session_with_metrics(
                            prompt, 
                            {
                                "response": response_text,
                                "context": f"Retrieved {len(docs)} documents" if 'docs' in locals() else "",
                                "response_time": 2.0,  # Placeholder
                                "token_count": len(response_text.split()),
                                "success": True,
                                "evaluations": []  # Will be populated by judge
                            },
                            selected_model,
                            metrics_collector
                        )
                    else:
                        # Create ChatMessage without evaluations
                        st.session_state["messages"].append(
                            ChatMessage(role="assistant", content=response_text)
                        )
                else:
                    error_msg = f"❌ Agent system error: {', '.join(result.errors)}"
                    st.markdown(error_msg)
                    st.session_state["messages"].append(
                        ChatMessage(role="assistant", content=error_msg)
                    )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}", icon = "⛔️")


def _format_agent_response(documents, original_query):
    """Format agent response from retrieved documents"""
    response_parts = []
    
    response_parts.append(f"**🤖 Agent Response for:** '{original_query}'")
    response_parts.append("")
    response_parts.append("**📚 Retrieved Information:**")
    
    for doc in documents:
        response_parts.append(f"**Document {doc['rank']}** (Page {doc.get('page_number', 'N/A')}):")
        response_parts.append(f"{doc['content']}")
        response_parts.append("---")
    
    response_parts.append("")
    response_parts.append("*💡 This is Phase 1 - more advanced analysis coming in Phase 2!*")
    
    return "\n".join(response_parts)


def render_delete_button_agent():
    """Delete button using agent system"""
    delete_collection = st.button(
        "⚠️ Delete collection", 
        type = "secondary",
        key = "delete_button"
    )

    if delete_collection:
        from processing.vector_db import delete_vector_db
        delete_vector_db(st.session_state["vector_db"])
        
        # Also clear agent vector DB
        if "agent_manager" in st.session_state:
            data_agent = st.session_state.agent_manager.get_agent("data_retriever")
            if data_agent:
                data_agent.clear_vector_db()


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