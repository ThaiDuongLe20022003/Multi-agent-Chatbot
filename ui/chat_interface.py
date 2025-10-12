"""
Chat interface components for the Streamlit application.
"""

import streamlit as st

from data_models.models import ChatMessage


def render_chat_interface(vector_db, selected_model, evaluation_enabled, judge_evaluator, metrics_collector, use_multi_agent):
    """Render the main chat interface"""
    message_container = st.container(height = 500, border = True)

    # Display chat history
    display_chat_history(message_container)

    # Chat input and processing
    if prompt := st.chat_input("Enter a prompt here...", key = "chat_input"):
        handle_user_input(prompt, message_container, vector_db, selected_model, 
                         evaluation_enabled, judge_evaluator, metrics_collector, use_multi_agent)


def display_chat_history(message_container):
    """Display the chat message history with multi-agent support"""
    for i, message in enumerate(st.session_state["messages"]):
        avatar = "🤖" if message.role == "assistant" else "😎"
        with message_container.chat_message(message.role, avatar = avatar):
            st.markdown(message.content)
            
            # Show multi-agent analyses if available
            if hasattr(message, 'agent_analyses') and message.agent_analyses:
                with st.expander("🔍 Multi-Agent Analysis Details"):
                    st.write("### Specialist Perspectives")
                    
                    for agent_analysis in message.agent_analyses:
                        st.write(f"**{agent_analysis['agent_type'].replace('_', ' ').title()}** "
                                f"(Confidence: {agent_analysis['confidence']:.0%})")
                        
                        with st.expander(f"View {agent_analysis['agent_type']} analysis"):
                            st.write(agent_analysis['analysis'])
                    
                    if hasattr(message, 'consensus_score'):
                        st.metric("Team Consensus Score", f"{message.consensus_score:.0%}")
                    
                    st.info("🤝 Agents collaborated through our horizontal multi-agent system")
            
            # Show evaluation scores if available
            if hasattr(message, 'evaluations') and message.evaluations:
                if isinstance(message.evaluations[0], dict):
                    avg_score = sum(eval_obj["overall_score"] for eval_obj in message.evaluations) / len(message.evaluations)
                else:
                    avg_score = sum(eval_obj.overall_score for eval_obj in message.evaluations) / len(message.evaluations)
                st.caption(f"📊 Average Evaluation: {avg_score:.1f}/10.0 ({len(message.evaluations)} judges)")


def handle_user_input(prompt, message_container, vector_db, selected_model, 
                     evaluation_enabled, judge_evaluator, metrics_collector, use_multi_agent):
    """Handle user input and generate response"""
    try:
        # Add user message to chat
        st.session_state["messages"].append(ChatMessage(role = "user", content = prompt))
        with message_container.chat_message("user", avatar = "😎"):
            st.markdown(prompt)

        # Process and display assistant response
        with message_container.chat_message("assistant", avatar = "🤖"):
            with st.spinner("Processing your question..."):
                if vector_db is not None:
                    import time
                    from processing.rag_chain import process_question_with_agents, process_question_simple, count_tokens
                    
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