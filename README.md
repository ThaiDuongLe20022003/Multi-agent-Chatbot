# DeepLaw RAG System - Multi-Agent Legal Analysis

## 🎯 Project Overview

**DeepLaw** is an advanced Legal Retrieval-Augmented Generation (RAG) system that provides AI-powered legal document analysis and question-answering. The system features a horizontal multi-agent architecture with full backward compatibility to single-agent operation.

## ✨ Key Features

### 🤖 Multi-Agent Legal Analysis
- **Specialist Legal Agents** with domain expertise
- **Horizontal Architecture** for agent collaboration
- **Consensus Building** that synthesizes multiple perspectives
- **Confidence Scoring** for each agent's analysis

### 📚 Legal Document Processing
- **PDF Document Analysis** with text extraction and embedding
- **Vector Database** for efficient legal document retrieval
- **Context-Aware Responses** based on uploaded legal documents
- **Multi-Page PDF Viewer** with zoom controls

### ⚖️ Quality Assurance
- **Multi-Judge Evaluation System** using multiple LLM models
- **Comprehensive Metrics** tracking response quality and performance
- **Real-time Evaluation** of faithfulness, relevance, and accuracy
- **Professional Reports** with detailed quality analysis

### 🎮 Dual Operation Modes
- **Single-Agent Mode**: Fast, focused legal analysis
- **Multi-Agent Mode**: Comprehensive multi-specialist analysis
- **Seamless Switching** between modes via UI toggle
- **Backward Compatibility** with all original features

## 🏗️ System Architecture

The DeepLaw system is built on a modular architecture that supports both single-agent and multi-agent operation modes:

**User Interface Layer**
- Streamlit-based web interface with real-time chat
- PDF document upload and viewer with zoom controls
- Sidebar configuration panel for model selection and settings
- Multi-agent analysis display with expandable sections

**Application Core Layer**
- Main orchestrator (main.py) managing the complete workflow
- Session state management for conversation history and document state
- Configuration management for models, paths, and application settings

**Processing Layer**
- Document Processing: PDF text extraction, chunking, and image rendering
- Vector Database: ChromaDB with HuggingFace embeddings for document storage
- RAG Pipeline: LangChain-based retrieval and generation with Ollama integration
- Multi-Agent Chain: Specialist agent coordination and consensus building

**Agent Layer**
- Base Legal Agent: Foundation class for all specialist agents
- Legal Research Agent: General legal document analysis and principles
- Case Law Agent: Precedent analysis and judicial reasoning
- Communication Hub: Horizontal agent coordination and message routing
- Consensus Builder: Synthesis of multiple agent perspectives

**Evaluation Layer**
- Multi-Judge Evaluator: Quality assessment using multiple LLM models
- Metrics Collector: Comprehensive performance tracking and analytics
- Metrics Storage: Persistent JSON-based evaluation data storage
- Report Generator: Professional evaluation reports and summaries

**Data Models Layer**
- LLM Evaluation: Faithfulness, relevance, completeness scoring
- LLM Metrics: Performance timing, token counts, throughput
- Chat Messages: Conversation history with evaluation data
- Agent Models: Agent analyses, communication messages, consensus data

## 🚀 Quick Start

### Installation

1. **Install Ollama**
   ```bash
   # On macOS/Linux
   curl -fsSL https://ollama.ai/install.sh | sh
   
   # On Windows, download from https://ollama.ai/download

   # Install Ollama models
   ollama pull <model_name>

2. **Clone the Repository**
   ```bash
   git clone https://github.com/ThaiDuongLe20022003/Multi-agent-Chatbot.git

3. **Install Python Dependencies**
   ```bash
    apt-get update
    apt install python3.10-venv
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

4. **Run the Application**
   ```bash
    streamlit run main.py