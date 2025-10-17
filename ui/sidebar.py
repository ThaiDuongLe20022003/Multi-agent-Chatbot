"""
Sidebar components for the Streamlit application.
Enhanced with horizontal multi-agent system statistics and visualization.
"""

import os
import json
import streamlit as st

from evaluation.evaluator import LLMJudgeEvaluator
from utils.helpers import extract_model_names
from config.settings import METRICS_DIR


def render_sidebar(available_models, selected_model, metrics_collector):
    """Render the sidebar with horizontal multi-agent system configuration"""
    st.header("⚙️ Horizontal System Configuration")
    
    # Model selection for response only
    if available_models:
        selected_model = st.selectbox(
            "Select Response Model", 
            available_models,
            key = "response_model_select",
            help = "Choose the model that will generate responses to your questions"
        )
    else:
        st.error("No Ollama models found. Please make sure Ollama is running.")
        st.stop()
    
    # Get judge models (all models except the selected one)
    judge_models = [model for model in available_models if model != selected_model]
    
    # Display judge models info for horizontal QA
    st.header("👨‍⚖️ Judge Models for QA")
    if judge_models:
        st.write(f"**{len(judge_models)} models** will evaluate each response:")
        for model in judge_models:
            st.write(f"• {model}")
        
        st.info("🤝 Quality Assurance Agent uses these models for unbiased evaluation")
    else:
        st.warning("No other models available for horizontal quality assurance")
    
    # Evaluation toggle
    evaluation_enabled = st.toggle(
        "Enable Multi-Judge Evaluation", 
        value = True,
        key = "eval_toggle",
        help = "Enable horizontal quality assurance with multiple judge models"
    )
    
    # Initialize multi-judge evaluator if needed
    judge_evaluator = None
    if judge_models and evaluation_enabled:
        judge_evaluator = LLMJudgeEvaluator(judge_models)
        st.session_state["judge_evaluator"] = judge_evaluator
    
    # Horizontal system metrics
    render_horizontal_system_metrics()
    
    # Traditional metrics actions
    st.header("📊 Evaluation Metrics")
    
    if st.button("📈 Show Evaluation Report"):
        report = metrics_collector.generate_report()
        st.text_area("Evaluation Report", report, height = 300)
    
    if st.button("💾 Save All Metrics to File"):
        if metrics_collector.current_session_metrics:
            filename = metrics_collector.save_all_metrics_to_file()
            if filename:
                st.success(f"All metrics saved to: {filename}")
            else:
                st.error("Failed to save metrics.")
        else:
            st.warning("No metrics to save.")
    
    if st.button("🗑️ Clear Metrics"):
        metrics_collector.current_session_metrics = []
        st.success("Metrics cleared.")
        
    # Saved metrics files section
    render_saved_metrics_section()
    
    return selected_model, evaluation_enabled, judge_evaluator


def render_horizontal_system_metrics():
    """Render horizontal multi-agent system metrics and statistics"""
    st.header("🤖 Horizontal System Analytics")
    
    if "agent_manager" in st.session_state and st.session_state.get("horizontal_system_active", False):
        agent_manager = st.session_state.agent_manager
        stats = agent_manager.get_workflow_stats()
        
        # System Overview Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Executions", stats["total_executions"])
        with col2:
            st.metric("Success Rate", f"{stats['success_rate']:.1%}")
        with col3:
            st.metric("Parallel Executions", stats["parallel_executions"])
        
        # Performance Metrics
        st.subheader("⏱️ Performance")
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        
        with perf_col1:
            st.metric("Avg Processing Time", f"{stats['average_processing_time']}s")
        with perf_col2:
            st.metric("Total Collaborations", stats["total_collaborations"])
        with perf_col3:
            cache_size = stats.get("cache_size", 0)
            st.metric("Cache Size", cache_size)
        
        # Agent Usage Breakdown
        if stats["agent_usage"]:
            st.subheader("👥 Agent Usage Distribution")
            total_executions = stats["total_executions"]
            
            for agent, count in sorted(stats["agent_usage"].items(), key = lambda x: x[1], reverse = True):
                usage_percent = (count / total_executions) * 100 if total_executions > 0 else 0
                progress_bar = st.progress(0)
                progress_bar.progress(min(usage_percent / 100, 1.0))
                st.write(f"**{agent}**: {count} uses ({usage_percent:.1f}%)")
        
        # Workflow Type Distribution
        if stats.get("workflow_types"):
            st.subheader("🔄 Workflow Type Usage")
            workflow_counts = {}
            for workflow in stats["workflow_types"]:
                workflow_counts[workflow] = workflow_counts.get(workflow, 0) + 1
            
            for workflow, count in sorted(workflow_counts.items(), key = lambda x: x[1], reverse = True):
                workflow_name = workflow.replace('_', ' ').title()
                st.write(f"• **{workflow_name}**: {count} times")
        
        # System Health Status
        st.subheader("🟢 System Health")
        health_status = agent_manager.health_check_all()
        
        health_col1, health_col2 = st.columns(2)
        healthy_agents = [name for name, healthy in health_status.items() if healthy]
        unhealthy_agents = [name for name, healthy in health_status.items() if not healthy]
        
        with health_col1:
            st.write(f"✅ **Healthy**: {len(healthy_agents)} agents")
            for agent in healthy_agents:
                st.write(f"  - {agent}")
        
        with health_col2:
            if unhealthy_agents:
                st.write(f"❌ **Unhealthy**: {len(unhealthy_agents)} agents")
                for agent in unhealthy_agents:
                    st.write(f"  - {agent}")
            else:
                st.write("🎉 All agents healthy!")
        
        # Collaboration Network Visualization
        st.subheader("🕸️ Collaboration Network")
        if stats["total_collaborations"] > 0:
            avg_collaborations = stats["total_collaborations"] / stats["total_executions"]
            st.write(f"**Average collaborations per query**: {avg_collaborations:.1f}")
            
            if stats["agent_usage"]:
                st.write("**Most active collaborators**:")
                active_agents = sorted(stats["agent_usage"].items(), key=lambda x: x[1], reverse=True)[:3]
                for agent, count in active_agents:
                    st.write(f"  - {agent}: {count} interactions")
        else:
            st.info("No collaboration data yet. Start chatting to see agent interactions!")
        
        # System Capabilities
        with st.expander("🔧 System Capabilities"):
            from agents import list_agent_capabilities
            capabilities = list_agent_capabilities()
            
            for agent_name, agent_caps in capabilities.items():
                st.write(f"### {agent_name}")
                st.write(f"**Role**: {agent_caps['primary_role']}")
                
                st.write("**Capabilities**:")
                for capability in agent_caps['capabilities']:
                    st.write(f"  - {capability}")
                
                st.write("**Horizontal Features**:")
                for feature in agent_caps['horizontal_features']:
                    st.write(f"  - {feature}")
                
                st.write("---")
    
    else:
        st.warning("Horizontal system not active or initialized")
        st.info("Upload a PDF and start chatting to activate the horizontal multi-agent system")


def render_saved_metrics_section():
    """Render the saved evaluation files section"""
    st.header("📁 Saved Evaluation Files")
    
    # Show list of evaluation files
    evaluation_files = []
    if os.path.exists(METRICS_DIR):
        evaluation_files = [f for f in os.listdir(METRICS_DIR) if f.endswith('.json') and f.startswith('evaluation_')]
        evaluation_files.sort(reverse=True)  # Newest first
    
    if evaluation_files:
        selected_file = st.selectbox("Select evaluation file", evaluation_files)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("👀 View Selected Evaluation"):
                filepath = os.path.join(METRICS_DIR, selected_file)
                try:
                    with open(filepath, 'r', encoding = 'utf-8') as f:
                        data = json.load(f)
                        st.json(data)
                except Exception as e:
                    st.error(f"Error reading file: {e}")
        
        with col2:
            if st.button("📥 Download Selected Evaluation"):
                filepath = os.path.join(METRICS_DIR, selected_file)
                try:
                    with open(filepath, 'r', encoding = 'utf-8') as f:
                        data = f.read()
                        st.download_button(
                            label = "Download JSON",
                            data = data,
                            file_name = selected_file,
                            mime = "application/json"
                        )
                except Exception as e:
                    st.error(f"Error reading file: {e}")
    else:
        st.info("No evaluation files found. Enable evaluation and start chatting to generate metrics.")


def render_metrics_summary(metrics_collector):
    """Render current session metrics summary with horizontal insights"""
    if metrics_collector.current_session_metrics:
        st.subheader("📈 Current Session Summary")
        summary = metrics_collector.get_session_summary()
        
        if summary:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Interactions", summary["total_interactions"])
                st.metric("Total Evaluations", summary["total_evaluations"])
            
            with col2:
                st.metric("Avg Response Time", f"{summary['avg_response_time']}s")
                if "avg_overall_score" in summary:
                    st.metric("Overall Quality", f"{summary['avg_overall_score']}/10.0")
            
            with col3:
                st.metric("Total Tokens", summary["total_tokens_generated"])
                st.metric("Avg Tokens/s", f"{summary['avg_tokens_per_second']:.1f}")
                
            # Display rating distribution as a bar chart
            if "rating_distribution" in summary:
                st.subheader("🎯 Rating Distribution")
                rating_data = summary["rating_distribution"]
                
                # Create a visual representation
                for rating, count in rating_data.items():
                    if count > 0:
                        percentage = (count / summary["total_evaluations"]) * 100
                        st.write(f"**{rating}**: {count} evaluations ({percentage:.1f}%)")
                        st.progress(min(percentage / 100, 1.0))
            
            # Judge model comparison
            if "judge_models" in summary:
                st.subheader("⚖️ Judge Model Comparison")
                for judge, stats in summary["judge_models"].items():
                    st.write(f"**{judge}**: {stats['avg_score']}/10.0 ({stats['count']} evaluations)")