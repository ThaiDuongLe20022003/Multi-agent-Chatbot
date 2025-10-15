"""
Streamlit application for PDF-based RAG using Multi-Agent System.
Enhanced with legal intelligence agents.
"""

import streamlit as st
import ollama
import warnings
import time

# Suppress torch warning
warnings.filterwarnings('ignore', category = UserWarning, message = '.*torch.classes.*')

# Import modular components
from config import STREAMLIT_CONFIG
from data_models import ChatMessage
from evaluation import MetricsCollector
from ui import render_sidebar, render_metrics_summary
from ui import render_pdf_uploader, render_pdf_viewer, render_delete_button
from utils import setup_logging, initialize_session_state, extract_model_names

# Import multi-agent system
from agents.agent_manager import AgentManager
from agents.pdf_processing_agent import PDFProcessingAgent
from agents.data_retrieval_agent import DataRetrievalAgent
from agents.legal_analyzer_agent import LegalAnalyzerAgent
from agents.summarize_reason_agent import SummarizeReasonAgent


def main():
    """Main application function"""
    # Setup application
    setup_logging()
    st.set_page_config(**STREAMLIT_CONFIG)
    initialize_session_state()  
    
    # Get available models
    available_models = get_available_models()
    
    # Initialize multi-agent system (will show error if no models)
    initialize_agent_system(available_models)
    
    # Initialize metrics collector
    if "metrics_collector" not in st.session_state:
        st.session_state["metrics_collector"] = MetricsCollector()
    
    # Create layout
    col1, col2 = st.columns([1.5, 2])
    
    # Render sidebar only if agent system is properly initialized
    with st.sidebar:
        # Only show model selection if we have models
        if available_models:
            selected_model, evaluation_enabled, judge_evaluator = render_sidebar(
                available_models, 
                st.session_state.get("selected_model", available_models[0]),
                st.session_state["metrics_collector"]
            )
            
            # Update selected model in session state
            st.session_state["selected_model"] = selected_model
            st.session_state["evaluation_enabled"] = evaluation_enabled
            
            # Update agent models if selection changed
            update_agent_models(selected_model)
            
            # Display agent system status
            render_agent_system_status()
            
            # Display metrics summary
            render_metrics_summary(st.session_state["metrics_collector"])
        else:
            st.error("No models available. Please install Ollama models.")
    
    # Main content - PDF upload and viewer (always show these)
    with col1:
        file_upload = render_pdf_uploader()
        if available_models:  # Only process PDF if models are available
            handle_pdf_upload_agent(file_upload)
        render_pdf_viewer()
        if available_models:  # Only show delete button if models are available
            render_delete_button_agent()
    
    # Main content - Chat interface
    with col2:
        if available_models:  # Only show chat if models are available
            render_chat_interface_agent(
                st.session_state["selected_model"],
                st.session_state["evaluation_enabled"],
                judge_evaluator,
                st.session_state["metrics_collector"]
            )
        else:
            st.info("💡 Install an Ollama model to enable chat functionality")


def initialize_agent_system(available_models):
    """Initialize the multi-agent system with Phase 2 agents"""
    if "agent_manager" not in st.session_state:
        st.session_state.agent_manager = AgentManager()
        
        # Register all agents
        pdf_agent = PDFProcessingAgent()
        data_agent = DataRetrievalAgent()
        legal_agent = LegalAnalyzerAgent()
        summarize_agent = SummarizeReasonAgent()
        
        st.session_state.agent_manager.register_agent(pdf_agent)
        st.session_state.agent_manager.register_agent(data_agent)
        st.session_state.agent_manager.register_agent(legal_agent)
        st.session_state.agent_manager.register_agent(summarize_agent)
        
        # If no Ollama models are available, inform the user and skip model initialization
        if not available_models:
            st.error(
                "No Ollama models detected. Please install an Ollama model locally and restart the app.\n\n"
                "Example (run in your terminal): `ollama pull llama2`\n\n"
                "After installing a model, refresh this app to use the agents with the installed model."
            )
            st.session_state.agent_system_initialized = False
            print("⚠️ Agent system initialized without a model because no Ollama models were found.")
            return

        # Initialize with the first available model
        default_model = available_models[0]
        update_agent_models(default_model)
        
        st.session_state.agent_system_initialized = True
        print(f"🤖 Multi-agent system initialized with 4 agents (Phase 2) using model: {default_model}")


def update_agent_models(selected_model: str):
    """Update all agents with the selected model"""
    if "agent_manager" not in st.session_state:
        return
        
    agent_manager = st.session_state.agent_manager
    
    # Update legal analyzer agent
    legal_agent = agent_manager.get_agent("legal_analyzer")
    if legal_agent:
        try:
            legal_agent.initialize_model(selected_model)
        except Exception as e:
            st.error(f"Failed to initialize legal analyzer with {selected_model}: {e}")
    
    # Update summarize reason agent
    summarize_agent = agent_manager.get_agent("summarize_reason")
    if summarize_agent:
        try:
            summarize_agent.initialize_model(selected_model)
        except Exception as e:
            st.error(f"Failed to initialize summarize reason agent with {selected_model}: {e}")


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
        
        # Agent usage
        if 'agent_usage' in stats and stats['agent_usage']:
            st.subheader("Agent Usage")
            for agent, count in stats['agent_usage'].items():
                st.write(f"• {agent}: {count} times")
        
        # Current model
        st.subheader("Current Model")
        st.write(f"🤖 {st.session_state.get('selected_model', 'Not set')}")
        
        # Phase 2 info
        st.subheader("Available Agents")
        st.write("• PDF Processing Agent")
        st.write("• Data Retrieval Agent") 
        st.write("• Legal Analyzer Agent 🆕")
        st.write("• Summarize & Reason Agent 🆕")
        st.success("Phase 2: Legal Intelligence Active")
        
        # Legal capabilities info
        with st.expander("Legal Analysis Capabilities"):
            st.write("""
            The system now provides:
            - **Legal Issue Identification**
            - **Legal Principles Application**  
            - **Case Law Interpretation**
            - **Comprehensive Legal Reasoning**
            - **Practical Implications Analysis**
            """)
    else:
        st.warning("Agent system not initialized")


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


def render_chat_interface_agent(selected_model, evaluation_enabled, judge_evaluator, metrics_collector):
    """Chat interface using agent system"""
    message_container = st.container(height = 500, border = True)

    # Display chat history
    display_chat_history(message_container)

    # Chat input and processing
    if prompt := st.chat_input("Ask about your PDF document...", key = "chat_input"):
        handle_user_input_agent(prompt, message_container, selected_model, 
                              evaluation_enabled, judge_evaluator, metrics_collector)


def display_chat_history(message_container):
    """Display the chat message history"""
    # Initialize messages if not exists
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message.role == "assistant" else "😎"
        with message_container.chat_message(message.role, avatar = avatar):
            st.markdown(message.content)
            
            # Show evaluation scores if available
            if message.evaluations:
                # Handle both dictionary and object formats
                if isinstance(message.evaluations[0], dict):
                    avg_score = sum(eval_obj["overall_score"] for eval_obj in message.evaluations) / len(message.evaluations)
                else:
                    avg_score = sum(eval_obj.overall_score for eval_obj in message.evaluations) / len(message.evaluations)
                st.caption(f"📊 Average Evaluation: {avg_score:.1f}/10.0 ({len(message.evaluations)} judges)")


def handle_user_input_agent(prompt, message_container, selected_model, 
                           evaluation_enabled, judge_evaluator, metrics_collector):
    """Handle user input using agent system"""
    try:
        # Add user message to chat
        st.session_state["messages"].append(ChatMessage(role = "user", content = prompt))
        
        with message_container.chat_message("user", avatar = "😎"):
            st.markdown(prompt)

        # Process with agent system
        with message_container.chat_message("assistant", avatar = "🤖"):
            with st.spinner("🤖 Agents are collaborating..."):
                start_time = time.time()
                
                agent_manager = st.session_state.agent_manager
                result = agent_manager.execute_workflow(prompt, "streamlit_session")
                
                response_time = time.time() - start_time
                
                if result.success:
                    st.markdown(result.final_response)
                    
                    # Record metrics if evaluation enabled
                    if evaluation_enabled and judge_evaluator:
                        update_session_with_metrics(
                            prompt, 
                            result.final_response,
                            response_time,
                            selected_model,
                            metrics_collector,
                            judge_evaluator
                        )
                    else:
                        # Create ChatMessage without evaluations
                        st.session_state["messages"].append(
                            ChatMessage(role = "assistant", content = result.final_response)
                        )
                        
                    # Show which agents were used
                    agents_used = list(result.agent_responses.keys())
                    if agents_used:
                        st.caption(f"🤖 Agents used: {', '.join(agents_used)}")
                        
                else:
                    error_msg = f"❌ Agent system error: {', '.join(result.errors)}"
                    st.markdown(error_msg)
                    st.session_state["messages"].append(
                        ChatMessage(role = "assistant", content = error_msg)
                    )

    except Exception as e:
        st.error(f"❌ Error: {str(e)}", icon = "⛔️")


def update_session_with_metrics(prompt, response, response_time, selected_model, metrics_collector, judge_evaluator):
    """Update session state with metrics and evaluations"""
    try:
        # Get context for evaluation (simplified for now)
        context = "Multi-agent legal analysis response"
        
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
        
        # Create ChatMessage object with evaluations
        st.session_state["messages"].append(
            ChatMessage(role = "assistant", content = response, evaluations = eval_dicts)
        )
        
    except Exception as e:
        st.error(f"Error recording metrics: {e}")
        # Fallback: create message without evaluations
        st.session_state["messages"].append(
            ChatMessage(role = "assistant", content = response)
        )


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
        
        st.rerun()


def get_available_models():
    """Get available Ollama models with basic error handling"""
    try:
        models_info = ollama.list()
        models = extract_model_names(models_info)
        return models
    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return tuple()  # Return empty tuple on error


if __name__ == "__main__":
    main()