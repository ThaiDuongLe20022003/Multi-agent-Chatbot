"""
Horizontal Multi-Agent Legal RAG System
Multiple specialized agents working collaboratively on legal document analysis
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Set protobuf environment variable to avoid error messages
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Define persistent directory for ChromaDB
PERSIST_DIRECTORY = os.path.join("data", "vectors")
METRICS_DIR = os.path.join("data", "metrics")

# Ensure directories exist
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Streamlit page configuration
st.set_page_config(
    page_title="DeepLaw - Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Define specialized agent roles"""
    RESEARCHER = "Legal Researcher"
    ANALYST = "Case Analyst"
    DRAFTER = "Document Drafter"
    REVIEWER = "Quality Reviewer"
    STRATEGIST = "Legal Strategist"


@dataclass
class AgentResponse:
    """Individual agent response with metadata"""
    role: AgentRole
    content: str
    confidence: float
    reasoning: str
    timestamp: str
    processing_time: float


@dataclass
class MultiAgentResponse:
    """Container for all agent responses"""
    query: str
    responses: Dict[AgentRole, AgentResponse]
    consensus_summary: str
    final_recommendation: str


class LegalResearcherAgent:
    """Specialized in finding relevant legal precedents and citations"""

    def __init__(self, model: str):
        self.model = model
        self.llm = ChatOllama(model=model, request_timeout=120.0)
        self.role = AgentRole.RESEARCHER

    def research_question(self, query: str, context: str) -> AgentResponse:
        start_time = time.time()

        prompt = ChatPromptTemplate.from_template("""
        You are a LEGAL RESEARCHER specializing in finding relevant precedents, statutes, and case law.

        LEGAL DOCUMENT CONTEXT:
        {context}

        RESEARCH QUERY: {query}

        Your task:
        1. Identify key legal concepts and terminology
        2. Find relevant sections in the document that address these concepts
        3. Note any citations, references, or legal frameworks mentioned
        4. Assess the completeness of the legal coverage

        Provide your research findings in this format:
        KEY FINDINGS: [Bullet points of relevant legal content]
        LEGAL CONCEPTS: [List of identified legal principles]
        CITATIONS: [Any case law, statutes, or references found]
        CONFIDENCE: [0-10 score based on how well the document covers the query]

        RESEARCH ANALYSIS:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "context": context})

        # Extract confidence score
        confidence = self._extract_confidence(response)

        return AgentResponse(
            role=self.role,
            content=response,
            confidence=confidence,
            reasoning="Comprehensive legal research and precedent analysis",
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response"""
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
        if confidence_match:
            return min(10.0, max(0.0, float(confidence_match.group(1))))
        return 7.0  # Default medium confidence


class CaseAnalystAgent:
    """Specialized in analyzing case facts, arguments, and weaknesses"""

    def __init__(self, model: str):
        self.model = model
        self.llm = ChatOllama(model=model, request_timeout=120.0)
        self.role = AgentRole.ANALYST

    def analyze_case(self, query: str, context: str) -> AgentResponse:
        start_time = time.time()

        prompt = ChatPromptTemplate.from_template("""
        You are a CASE ANALYST specializing in legal argument analysis and weakness identification.

        LEGAL DOCUMENT CONTEXT:
        {context}

        ANALYSIS QUERY: {query}

        Your task:
        1. Analyze the strength of legal arguments presented
        2. Identify potential weaknesses or gaps in reasoning
        3. Evaluate factual sufficiency and evidence support
        4. Assess logical consistency and legal soundness

        Provide your analysis in this format:
        ARGUMENT STRENGTH: [Assessment of legal arguments]
        WEAKNESSES IDENTIFIED: [List of potential issues or gaps]
        EVIDENCE ASSESSMENT: [Evaluation of supporting facts]
        LOGICAL CONSISTENCY: [Analysis of reasoning flow]
        CONFIDENCE: [0-10 score based on argument quality]

        CASE ANALYSIS:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "context": context})

        confidence = self._extract_confidence(response)

        return AgentResponse(
            role=self.role,
            content=response,
            confidence=confidence,
            reasoning="Critical analysis of legal arguments and case weaknesses",
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )

    def _extract_confidence(self, response: str) -> float:
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
        if confidence_match:
            return min(10.0, max(0.0, float(confidence_match.group(1))))
        return 7.0


class DocumentDrafterAgent:
    """Specialized in drafting legal documents and responses"""

    def __init__(self, model: str):
        self.model = model
        self.llm = ChatOllama(model=model, request_timeout=120.0)
        self.role = AgentRole.DRAFTER

    def draft_response(self, query: str, context: str) -> AgentResponse:
        start_time = time.time()

        prompt = ChatPromptTemplate.from_template("""
        You are a DOCUMENT DRAFTER specializing in creating precise legal language and structured responses.

        LEGAL DOCUMENT CONTEXT:
        {context}

        DRAFTING QUERY: {query}

        Your task:
        1. Create well-structured legal responses
        2. Use precise legal terminology and formal language
        3. Organize information logically and clearly
        4. Ensure professional legal formatting

        Provide your drafted response in this format:
        DRAFTED RESPONSE: [Professional legal language answering the query]
        STRUCTURE: [Organization of the response]
        LEGAL PRECISION: [Assessment of terminology accuracy]
        CONFIDENCE: [0-10 score based on drafting quality]

        LEGAL DRAFT:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"query": query, "context": context})

        confidence = self._extract_confidence(response)

        return AgentResponse(
            role=self.role,
            content=response,
            confidence=confidence,
            reasoning="Professional legal drafting and structured response creation",
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )

    def _extract_confidence(self, response: str) -> float:
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
        if confidence_match:
            return min(10.0, max(0.0, float(confidence_match.group(1))))
        return 8.0  # Default higher confidence for drafting


class QualityReviewerAgent:
    """Specialized in quality control and validation"""

    def __init__(self, model: str):
        self.model = model
        self.llm = ChatOllama(model=model, request_timeout=120.0)
        self.role = AgentRole.REVIEWER

    def review_quality(self, query: str, context: str,
                       other_responses: Dict[AgentRole, AgentResponse]) -> AgentResponse:
        start_time = time.time()

        # Summarize other agent responses for review
        responses_summary = "\n".join([
            f"{role.value}: {response.content[:500]}..."
            for role, response in other_responses.items()
        ])

        prompt = ChatPromptTemplate.from_template("""
        You are a QUALITY REVIEWER specializing in legal document validation and quality assurance.

        LEGAL DOCUMENT CONTEXT:
        {context}

        ORIGINAL QUERY: {query}

        OTHER AGENT RESPONSES:
        {responses_summary}

        Your task:
        1. Validate accuracy and completeness of all responses
        2. Identify inconsistencies between different analyses
        3. Check for legal compliance and proper citation
        4. Assess overall quality and reliability

        Provide your review in this format:
        QUALITY ASSESSMENT: [Overall quality evaluation]
        INCONSISTENCIES: [Any conflicting information found]
        VALIDATION NOTES: [Accuracy and compliance check]
        IMPROVEMENT SUGGESTIONS: [Recommendations for enhancement]
        CONFIDENCE: [0-10 score based on overall quality]

        QUALITY REVIEW:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "context": context,
            "responses_summary": responses_summary
        })

        confidence = self._extract_confidence(response)

        return AgentResponse(
            role=self.role,
            content=response,
            confidence=confidence,
            reasoning="Comprehensive quality validation and consistency checking",
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )

    def _extract_confidence(self, response: str) -> float:
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
        if confidence_match:
            return min(10.0, max(0.0, float(confidence_match.group(1))))
        return 7.0


class LegalStrategistAgent:
    """Specialized in strategic legal advice and recommendations"""

    def __init__(self, model: str):
        self.model = model
        self.llm = ChatOllama(model=model, request_timeout=120.0)
        self.role = AgentRole.STRATEGIST

    def provide_strategy(self, query: str, context: str,
                         all_responses: Dict[AgentRole, AgentResponse]) -> AgentResponse:
        start_time = time.time()

        # Create comprehensive analysis summary
        analysis_summary = "\n".join([
            f"{role.value} (Confidence: {response.confidence}/10): {response.content[:300]}..."
            for role, response in all_responses.items()
        ])

        prompt = ChatPromptTemplate.from_template("""
        You are a LEGAL STRATEGIST specializing in strategic advice and actionable recommendations.

        LEGAL DOCUMENT CONTEXT:
        {context}

        CLIENT QUERY: {query}

        COMPREHENSIVE ANALYSIS:
        {analysis_summary}

        Your task:
        1. Synthesize all analyses into strategic recommendations
        2. Provide actionable next steps and considerations
        3. Assess risks and opportunities
        4. Offer practical legal strategy

        Provide your strategic advice in this format:
        STRATEGIC ASSESSMENT: [Overall strategic position]
        KEY RECOMMENDATIONS: [Actionable advice]
        RISK ANALYSIS: [Potential risks and mitigation]
        NEXT STEPS: [Concrete actions to take]
        CONFIDENCE: [0-10 score based on strategic soundness]

        STRATEGIC ADVICE:
        """)

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({
            "query": query,
            "context": context,
            "analysis_summary": analysis_summary
        })

        confidence = self._extract_confidence(response)

        return AgentResponse(
            role=self.role,
            content=response,
            confidence=confidence,
            reasoning="Strategic synthesis and actionable recommendations",
            timestamp=datetime.now().isoformat(),
            processing_time=time.time() - start_time
        )

    def _extract_confidence(self, response: str) -> float:
        confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', response)
        if confidence_match:
            return min(10.0, max(0.0, float(confidence_match.group(1))))
        return 7.0


class MultiAgentLegalSystem:
    """Orchestrates multiple specialized legal agents"""

    def __init__(self, vector_db, model_mapping: Dict[AgentRole, str]):
        self.vector_db = vector_db
        self.agents = {}

        # Initialize specialized agents
        self.agents[AgentRole.RESEARCHER] = LegalResearcherAgent(model_mapping.get(AgentRole.RESEARCHER, "llama2"))
        self.agents[AgentRole.ANALYST] = CaseAnalystAgent(model_mapping.get(AgentRole.ANALYST, "llama2"))
        self.agents[AgentRole.DRAFTER] = DocumentDrafterAgent(model_mapping.get(AgentRole.DRAFTER, "llama2"))
        self.agents[AgentRole.REVIEWER] = QualityReviewerAgent(model_mapping.get(AgentRole.REVIEWER, "llama2"))
        self.agents[AgentRole.STRATEGIST] = LegalStrategistAgent(model_mapping.get(AgentRole.STRATEGIST, "llama2"))

    def process_query(self, query: str) -> MultiAgentResponse:
        """Process query through all specialized agents"""
        # Retrieve relevant context
        retriever = self.vector_db.as_retriever(search_kwargs={"k": 4})
        context_docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        responses = {}

        # Phase 1: Parallel analysis by researcher, analyst, and drafter
        with st.spinner("🕵️ Conducting legal research..."):
            responses[AgentRole.RESEARCHER] = self.agents[AgentRole.RESEARCHER].research_question(query, context)

        with st.spinner("🔍 Analyzing case arguments..."):
            responses[AgentRole.ANALYST] = self.agents[AgentRole.ANALYST].analyze_case(query, context)

        with st.spinner("📝 Drafting legal response..."):
            responses[AgentRole.DRAFTER] = self.agents[AgentRole.DRAFTER].draft_response(query, context)

        # Phase 2: Quality review based on initial analyses
        with st.spinner("✅ Reviewing quality and consistency..."):
            responses[AgentRole.REVIEWER] = self.agents[AgentRole.REVIEWER].review_quality(query, context, responses)

        # Phase 3: Strategic synthesis
        with st.spinner("🎯 Developing legal strategy..."):
            responses[AgentRole.STRATEGIST] = self.agents[AgentRole.STRATEGIST].provide_strategy(query, context,
                                                                                                 responses)

        # Generate consensus and final recommendation
        consensus_summary = self._generate_consensus_summary(responses)
        final_recommendation = self._generate_final_recommendation(responses)

        return MultiAgentResponse(
            query=query,
            responses=responses,
            consensus_summary=consensus_summary,
            final_recommendation=final_recommendation
        )

    def _generate_consensus_summary(self, responses: Dict[AgentRole, AgentResponse]) -> str:
        """Generate summary of agent consensus"""
        avg_confidence = sum(response.confidence for response in responses.values()) / len(responses)

        summary = f"## Multi-Agent Analysis Summary\n\n"
        summary += f"**Overall Confidence: {avg_confidence:.1f}/10.0**\n\n"

        for role, response in responses.items():
            summary += f"**{role.value}**: {response.confidence}/10 confidence\n"

        return summary

    def _generate_final_recommendation(self, responses: Dict[AgentRole, AgentResponse]) -> str:
        """Generate final synthesized recommendation"""
        strategist_response = responses[AgentRole.STRATEGIST].content

        # Extract key recommendations from strategist
        recommendations_match = re.search(r'KEY RECOMMENDATIONS:\s*(.*?)(?=\n\w+:|$)', strategist_response, re.DOTALL)
        next_steps_match = re.search(r'NEXT STEPS:\s*(.*?)(?=\n\w+:|$)', strategist_response, re.DOTALL)

        final_rec = "## 🎯 Final Legal Recommendation\n\n"

        if recommendations_match:
            final_rec += "### Key Recommendations:\n" + recommendations_match.group(1).strip() + "\n\n"

        if next_steps_match:
            final_rec += "### Next Steps:\n" + next_steps_match.group(1).strip()

        return final_rec


# ========== UTILITY FUNCTIONS FROM ORIGINAL CODE ==========

def extract_model_names(models_info: Any) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.
    """
    logger.info("Extracting model names from models_info")
    try:
        if hasattr(models_info, "models"):
            model_names = tuple(model.model for model in models_info.models)
        else:
            model_names = tuple()

        logger.info(f"Extracted model names: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Error extracting model names: {e}")
        return tuple()


def create_simple_vector_db(file_upload) -> Chroma:
    """Create a simple vector DB without complex embeddings to avoid the meta tensor error"""
    logger.info(f"Creating simple vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    try:
        path = os.path.join(temp_dir, file_upload.name)
        with open(path, "wb") as f:
            f.write(file_upload.getvalue())

        # Use PyPDFLoader for simplicity
        loader = PyPDFLoader(path)
        data = loader.load_and_split()

        # Simple text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(data)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Use a simpler embedding model that doesn't cause the meta tensor issue
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Smaller, more compatible model
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}  # Simpler configuration
        )

        # Create vector store
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY,
            collection_name=f"pdf_{hash(file_upload.name)}"
        )

        logger.info("Simple vector DB created successfully")
        return vector_db

    except Exception as e:
        logger.error(f"Error creating vector DB: {e}")
        st.error(f"Error creating vector database: {str(e)}")
        raise
    finally:
        shutil.rmtree(temp_dir)


def get_simple_retriever(vector_db: Chroma):
    """Create a simple retriever without complex query expansion"""
    # Simple retriever configuration
    retriever = vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve 4 most similar documents
    )
    return retriever


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.
    """
    logger.info(f"Extracting all pages as images from file: {file_upload.name}")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        try:
            vector_db.delete_collection()

            st.session_state.pop("pdf_pages", None)
            st.session_state.pop("file_upload", None)
            st.session_state.pop("vector_db", None)

            st.success("Collection and temporary files deleted successfully.")
            logger.info("Vector DB and related session state cleared")
            st.rerun()
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")
            logger.error(f"Error deleting collection: {e}")
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def count_tokens(text: str) -> int:
    """Simple token counter"""
    return len(text.split())


def get_agent_specialization(role: AgentRole) -> str:
    """Get agent specialization description"""
    specializations = {
        AgentRole.RESEARCHER: "Legal precedents, statutes, and case law research",
        AgentRole.ANALYST: "Argument strength analysis and weakness identification",
        AgentRole.DRAFTER: "Precise legal language and document drafting",
        AgentRole.REVIEWER: "Quality validation and consistency checking",
        AgentRole.STRATEGIST: "Strategic advice and actionable recommendations"
    }
    return specializations.get(role, "Legal analysis")


def get_agent_focus(role: AgentRole) -> str:
    """Get agent focus description"""
    focuses = {
        AgentRole.RESEARCHER: "Finding relevant legal authorities and citations",
        AgentRole.ANALYST: "Critical evaluation of legal arguments and evidence",
        AgentRole.DRAFTER: "Creating professional, well-structured legal responses",
        AgentRole.REVIEWER: "Ensuring accuracy, completeness, and compliance",
        AgentRole.STRATEGIST: "Synthesizing analyses into practical legal strategy"
    }
    return focuses.get(role, "Legal analysis")


def main():
    st.title("🤖 DeepLaw - Multi-Agent Legal Analysis System")

    # Get available models
    try:
        models_info = ollama.list()
        available_models = extract_model_names(models_info)
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        available_models = tuple()
        st.stop()

    # Initialize session state
    if "multi_agent_messages" not in st.session_state:
        st.session_state["multi_agent_messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None
    if "pdf_pages" not in st.session_state:
        st.session_state["pdf_pages"] = None

    # Layout
    col1, col2 = st.columns([1, 2])

    # Sidebar for agent configuration
    with st.sidebar:
        st.header("⚙️ Agent Configuration")

        if available_models:
            # Model assignment for each agent
            st.subheader("Assign Models to Agents")
            model_mapping = {}
            for role in AgentRole:
                model_mapping[role] = st.selectbox(
                    f"{role.value} Model",
                    available_models,
                    key=f"model_{role.name}",
                    help=f"Select model for {role.value}"
                )
        else:
            st.error("No Ollama models available")
            st.stop()

        st.header("📊 Agent Roles Overview")
        for role in AgentRole:
            with st.expander(f"{role.value}"):
                st.write(f"**Specialization**: {get_agent_specialization(role)}")
                st.write(f"**Primary Focus**: {get_agent_focus(role)}")

        # PDF viewer controls
        if st.session_state.get("pdf_pages"):
            st.header("📄 Document Viewer")
            if st.button("Clear Document"):
                delete_vector_db(st.session_state["vector_db"])

    # Main content area
    with col1:
        st.header("📁 Document Management")
        file_upload = st.file_uploader(
            "Upload Legal PDF Document",
            type="pdf",
            key="pdf_uploader"
        )

        if file_upload and st.session_state["vector_db"] is None:
            with st.spinner("Processing legal document..."):
                try:
                    st.session_state["vector_db"] = create_simple_vector_db(file_upload)
                    # Extract PDF pages for display
                    with pdfplumber.open(file_upload) as pdf:
                        st.session_state["pdf_pages"] = [page.to_image().original for page in pdf.pages]
                    st.success("Document processed! Agents are ready.")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

        # Display PDF pages if available
        if st.session_state.get("pdf_pages"):
            st.subheader("Document Preview")
            zoom_level = st.slider("Zoom Level", 100, 500, 300, key="zoom_slider")
            with st.container(height=400, border=True):
                for page_image in st.session_state["pdf_pages"][:3]:  # Show first 3 pages
                    st.image(page_image, width=zoom_level)

    # Chat interface
    with col2:
        st.header("💬 Multi-Agent Legal Consultation")

        # Display chat history
        chat_container = st.container(height=600, border=True)

        with chat_container:
            for message in st.session_state["multi_agent_messages"]:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])
                else:
                    # Display multi-agent response
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown("### 🤖 Multi-Agent Analysis Complete")

                        # Show final recommendation first
                        if "final_recommendation" in message:
                            st.markdown(message["final_recommendation"])

                        # Show consensus summary
                        if "content" in message and message["content"]:
                            with st.expander("📊 Analysis Summary", expanded=False):
                                st.markdown(message["content"])

                        # Agent responses in expanders
                        if "responses" in message:
                            for role_name, response in message["responses"].items():
                                role = AgentRole[role_name]
                                with st.expander(f"**{role.value}** (Confidence: {response['confidence']}/10)",
                                                 expanded=False):
                                    st.markdown(f"**Processing Time**: {response['processing_time']:.2f}s")
                                    st.markdown(f"**Reasoning**: {response['reasoning']}")
                                    st.markdown("**Analysis**:")
                                    st.markdown(response['content'])

        # Chat input
        if prompt := st.chat_input("Ask a legal question..."):
            if st.session_state["vector_db"] is None:
                st.warning("Please upload a legal document first.")
            else:
                # Add user message
                st.session_state["multi_agent_messages"].append({
                    "role": "user",
                    "content": prompt
                })

                # Process with multi-agent system
                with st.spinner("🤖 Assembling legal team for analysis..."):
                    multi_agent_system = MultiAgentLegalSystem(
                        st.session_state["vector_db"],
                        model_mapping
                    )

                    result = multi_agent_system.process_query(prompt)

                # Convert responses to serializable format
                serializable_responses = {}
                for role, response in result.responses.items():
                    serializable_responses[role.name] = {
                        "role": role.value,
                        "content": response.content,
                        "confidence": response.confidence,
                        "reasoning": response.reasoning,
                        "timestamp": response.timestamp,
                        "processing_time": response.processing_time
                    }

                # Add assistant message
                st.session_state["multi_agent_messages"].append({
                    "role": "assistant",
                    "content": result.consensus_summary,
                    "responses": serializable_responses,
                    "final_recommendation": result.final_recommendation
                })

                st.rerun()


if __name__ == "__main__":
    main()