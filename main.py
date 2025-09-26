"""
Horizontal Multi-Agent Chatbot System for Legal RAG (Retrieval-Augmented Generation)

SYSTEM OVERVIEW:
- Type: Horizontal (Peer-to-Peer) Multi-Agent Architecture
- Communication: Asynchronous message-passing between specialized agents
- Coordination: Orchestration agent manages workflow and inter-agent communication
- Scalability: Agents can be added/removed without central bottleneck

AGENT HIERARCHY & RESPONSIBILITIES:
1. Document Processing Agent: Handles PDF ingestion, text extraction, and vector database creation
2. Retrieval Agent: Performs semantic search on vector database to find relevant context
3. Query Understanding Agent: Analyzes user intent, query type, and complexity
4. Response Generation Agent: Generates answers using LLM with retrieved context
5. Multi-Judge Evaluation Agent: Evaluates response quality using multiple LLM judges
6. Orchestration Agent: Coordinates workflow and manages inter-agent communication

WORKFLOW:
User Query → Query Understanding → Document Retrieval → Response Generation → Multi-Judge Evaluation → Final Response

KEY FEATURES:
- Parallel processing capabilities
- Fault isolation between agents
- Scalable architecture
- Real-time agent monitoring
- Comprehensive evaluation system
"""

import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import warnings
import json
import re
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# Configuration
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Directory setup
PERSIST_DIRECTORY = os.path.join("data", "vectors")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Streamlit page configuration
st.set_page_config(
    page_title="DeepLaw Multi-Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Enumeration of all agent types"""
    DOCUMENT_PROCESSOR = "document_processor"
    RETRIEVAL = "retrieval"
    RESPONSE_GENERATOR = "response_generator"
    EVALUATOR = "evaluator"
    QUERY_UNDERSTANDING = "query_understanding"
    ORCHESTRATION = "orchestration"


@dataclass
class ProcessingResult:
    """Standardized result format for agent operations"""
    success: bool
    data: Any
    error: Optional[str] = None
    processing_time: float = 0.0


@dataclass
class LLMEvaluation:
    """LLM evaluation metrics"""
    faithfulness: float
    groundedness: float
    factual_consistency: float
    relevance: float
    completeness: float
    fluency: float
    overall_score: float
    evaluation_notes: str
    judge_model: str


class BaseAgent:
    """Base class for all agents"""

    def __init__(self, agent_type: AgentType, name: str):
        self.agent_type = agent_type
        self.name = name

    async def process(self, content: Dict) -> ProcessingResult:
        """Process content - to be implemented by subclasses"""
        raise NotImplementedError


class DocumentProcessingAgent(BaseAgent):
    """Handles PDF processing and vector database creation"""

    def __init__(self):
        super().__init__(AgentType.DOCUMENT_PROCESSOR, "Document Processor")
        self.vector_dbs = {}

    async def process(self, content: Dict) -> ProcessingResult:
        start_time = time.time()
        try:
            if content.get("action") == "process_pdf":
                file_upload = content["file_upload"]
                vector_db = self._create_vector_db(file_upload)
                pdf_pages = self._extract_pdf_pages(file_upload)

                return ProcessingResult(
                    success=True,
                    data={"vector_db": vector_db, "pdf_pages": pdf_pages},
                    processing_time=time.time() - start_time
                )
        except Exception as e:
            return ProcessingResult(success=False, error=str(e), processing_time=time.time() - start_time)

    def _create_vector_db(self, file_upload) -> Chroma:
        """Create vector database from PDF file"""
        temp_dir = tempfile.mkdtemp()
        try:
            # Save uploaded file temporarily
            path = os.path.join(temp_dir, file_upload.name)
            with open(path, "wb") as f:
                f.write(file_upload.getvalue())

            # Load and split PDF document
            loader = PyPDFLoader(path)
            data = loader.load_and_split()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200, length_function=len
            )
            chunks = text_splitter.split_documents(data)

            # Initialize embeddings model
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )

            # Create vector store
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY,
                collection_name=f"pdf_{hash(file_upload.name)}"
            )

            self.vector_dbs[file_upload.name] = vector_db
            return vector_db

        finally:
            shutil.rmtree(temp_dir)

    def _extract_pdf_pages(self, file_upload) -> List[Any]:
        """Extract PDF pages as images for display"""
        try:
            # Reset file pointer
            file_upload.seek(0)
            with pdfplumber.open(file_upload) as pdf:
                pages = []
                for page in pdf.pages:
                    try:
                        page_image = page.to_image()
                        pages.append(page_image)
                    except Exception as e:
                        logger.warning(f"Could not convert page to image: {e}")
                        # Add placeholder for pages that can't be converted
                        pages.append(None)
            return pages
        except Exception as e:
            logger.error(f"Error extracting PDF pages: {e}")
            return []


class RetrievalAgent(BaseAgent):
    """Performs semantic search on vector databases"""

    def __init__(self):
        super().__init__(AgentType.RETRIEVAL, "Retrieval Agent")

    async def process(self, content: Dict) -> ProcessingResult:
        start_time = time.time()
        try:
            if content.get("action") == "retrieve":
                query = content["query"]
                vector_db = content["vector_db"]

                # Configure retriever
                retriever = vector_db.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )

                # Retrieve relevant documents
                context_docs = retriever.invoke(query)
                context = "\n\n".join([
                    f"Document {i + 1}: {doc.page_content[:500]}..."
                    for i, doc in enumerate(context_docs[:3])
                ])

                return ProcessingResult(
                    success=True,
                    data={"context": context, "documents": context_docs},
                    processing_time=time.time() - start_time
                )
        except Exception as e:
            return ProcessingResult(success=False, error=str(e), processing_time=time.time() - start_time)


class ResponseGenerationAgent(BaseAgent):
    """Generates responses using LLM with retrieved context"""

    def __init__(self):
        super().__init__(AgentType.RESPONSE_GENERATOR, "Response Generator")
        self.model_name = self._get_available_model()
        self.prompt_template = """You are a professional legal expert. 

        CONTEXT INFORMATION:
        {context}

        QUESTION: {question}

        Please provide a helpful answer based on the context above. 
        If you cannot find the answer in the context, say so.

        ANSWER:
        """

    def _get_available_model(self) -> str:
        """Get the first available Ollama model"""
        try:
            models_info = ollama.list()
            if hasattr(models_info, 'models') and models_info.models:
                return models_info.models[0].model
            else:
                return "llama2"
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return "llama2"

    async def process(self, content: Dict) -> ProcessingResult:
        start_time = time.time()
        try:
            if content.get("action") == "generate_response":
                query = content["query"]
                context = content["context"]

                # Initialize LLM
                llm = ChatOllama(model=self.model_name, request_timeout=120.0)
                prompt = ChatPromptTemplate.from_template(self.prompt_template)

                # Create processing chain
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": query})

                return ProcessingResult(
                    success=True,
                    data={"response": response},
                    processing_time=time.time() - start_time
                )
        except Exception as e:
            return ProcessingResult(success=False, error=str(e), processing_time=time.time() - start_time)


class MultiJudgeEvaluationAgent(BaseAgent):
    """Evaluates response quality using multiple LLM judges"""

    def __init__(self):
        super().__init__(AgentType.EVALUATOR, "Multi-Judge Evaluator")
        self.judge_models = self._get_judge_models()
        self.evaluation_prompt = """Evaluate this response based on context and query:

        QUERY: {query}
        CONTEXT: {context}
        RESPONSE: {response}

        Provide scores (0.0-10.0) for:
        - faithfulness: reliance on context
        - relevance: addresses query  
        - completeness: covers aspects
        - fluency: natural language
        - overall_score: weighted average

        Respond with JSON only:
        {{
            "faithfulness": 8.5,
            "relevance": 9.0,
            "completeness": 7.5, 
            "fluency": 9.0,
            "overall_score": 8.5,
            "evaluation_notes": "Brief explanation"
        }}"""

    def _get_judge_models(self) -> List[str]:
        """Get all available models except the response model"""
        try:
            models_info = ollama.list()
            if hasattr(models_info, 'models') and models_info.models:
                all_models = [model.model for model in models_info.models]
                return all_models[1:] if len(all_models) > 1 else all_models
            return []
        except Exception as e:
            logger.error(f"Error getting judge models: {e}")
            return []

    async def process(self, content: Dict) -> ProcessingResult:
        start_time = time.time()
        try:
            if content.get("action") == "evaluate":
                query = content["query"]
                response = content["response"]
                context = content["context"]

                # Parallel evaluation
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for judge_model in self.judge_models:
                        future = executor.submit(
                            self._evaluate_single_judge,
                            query, response, context, judge_model
                        )
                        futures.append(future)

                    evaluations = []
                    for future in as_completed(futures):
                        try:
                            evaluations.append(future.result())
                        except Exception as e:
                            logger.error(f"Evaluation failed: {e}")

                return ProcessingResult(
                    success=True,
                    data={"evaluations": evaluations},
                    processing_time=time.time() - start_time
                )
        except Exception as e:
            return ProcessingResult(success=False, error=str(e), processing_time=time.time() - start_time)

    def _evaluate_single_judge(self, query: str, response: str, context: str, judge_model: str) -> Dict:
        """Evaluate response using a single judge model"""
        try:
            judge_llm = ChatOllama(model=judge_model, request_timeout=60.0)

            prompt = self.evaluation_prompt.format(
                query=query,
                context=context[:2000],
                response=response
            )

            evaluation_response = judge_llm.invoke(prompt)
            eval_text = evaluation_response.content.strip()

            # Parse JSON response
            json_match = re.search(r'\{.*\}', eval_text, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                eval_data = self._create_fallback_evaluation()

            eval_data['judge_model'] = judge_model
            return eval_data

        except Exception as e:
            logger.error(f"Evaluation error from {judge_model}: {e}")
            return self._create_fallback_evaluation(judge_model)

    def _create_fallback_evaluation(self, judge_model: str = "fallback") -> Dict:
        """Create fallback evaluation when evaluation fails"""
        return {
            "faithfulness": 5.0,
            "relevance": 5.0,
            "completeness": 5.0,
            "fluency": 6.0,
            "overall_score": 5.2,
            "evaluation_notes": "Evaluation unavailable",
            "judge_model": judge_model
        }


class QueryUnderstandingAgent(BaseAgent):
    """Analyzes user queries to understand intent"""

    def __init__(self):
        super().__init__(AgentType.QUERY_UNDERSTANDING, "Query Understanding Agent")

    async def process(self, content: Dict) -> ProcessingResult:
        start_time = time.time()
        try:
            if content.get("action") == "analyze_query":
                query = content["query"]

                # Simple query analysis
                analysis = self._analyze_query(query)

                return ProcessingResult(
                    success=True,
                    data=analysis,
                    processing_time=time.time() - start_time
                )
        except Exception as e:
            return ProcessingResult(success=False, error=str(e), processing_time=time.time() - start_time)

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query for intent and complexity"""
        query_lower = query.lower()

        # Simple classification
        if any(word in query_lower for word in ["what is", "define", "meaning"]):
            query_type = "definition"
        elif any(word in query_lower for word in ["how to", "procedure", "steps"]):
            query_type = "procedural"
        elif any(word in query_lower for word in ["compare", "difference", "vs"]):
            query_type = "comparative"
        elif any(word in query_lower for word in ["analyze", "why", "explain"]):
            query_type = "analytical"
        else:
            query_type = "factual"

        # Complexity assessment
        word_count = len(query.split())
        if word_count > 15:
            complexity = "complex"
        elif word_count > 8:
            complexity = "moderate"
        else:
            complexity = "simple"

        return {
            "query_type": query_type,
            "complexity": complexity,
            "key_entities": ["legal terms"],
            "intent": "Seeking legal information"
        }


class OrchestrationAgent(BaseAgent):
    """Coordinates workflow between all agents"""

    def __init__(self, agents: Dict[AgentType, BaseAgent]):
        super().__init__(AgentType.ORCHESTRATION, "Orchestration Agent")
        self.agents = agents

    async def process_user_query(self, query: str, file_upload=None) -> Dict[str, Any]:
        """End-to-end processing of user query"""
        start_time = time.time()

        try:
            # Step 1: Query Understanding
            query_analysis_result = await self.agents[AgentType.QUERY_UNDERSTANDING].process(
                {"action": "analyze_query", "query": query}
            )
            query_analysis = query_analysis_result.data if query_analysis_result.success else {}

            # Step 2: Document Processing (if file provided)
            vector_db = None
            pdf_pages = []
            if file_upload:
                doc_result = await self.agents[AgentType.DOCUMENT_PROCESSOR].process(
                    {"action": "process_pdf", "file_upload": file_upload}
                )
                if doc_result.success:
                    vector_db = doc_result.data["vector_db"]
                    pdf_pages = doc_result.data["pdf_pages"]

            # Step 3: Retrieval
            retrieval_result = await self.agents[AgentType.RETRIEVAL].process(
                {"action": "retrieve", "query": query, "vector_db": vector_db}
            )
            context = retrieval_result.data["context"] if retrieval_result.success else "No context available"

            # Step 4: Response Generation
            response_result = await self.agents[AgentType.RESPONSE_GENERATOR].process(
                {"action": "generate_response", "query": query, "context": context}
            )
            response = response_result.data["response"] if response_result.success else "Error generating response"

            # Step 5: Evaluation
            evaluation_result = await self.agents[AgentType.EVALUATOR].process(
                {"action": "evaluate", "query": query, "response": response, "context": context}
            )
            evaluations = evaluation_result.data["evaluations"] if evaluation_result.success else []

            # Compile final result
            return {
                "query": query,
                "response": response,
                "context": context,
                "query_analysis": query_analysis,
                "evaluations": evaluations,
                "pdf_pages": pdf_pages,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            return {
                "error": str(e),
                "query": query,
                "response": "System error occurred",
                "processing_time": time.time() - start_time
            }


class HorizontalMultiAgentSystem:
    """Main multi-agent system coordinating all specialized agents"""

    def __init__(self):
        self.agents = {}
        self.orchestrator = None
        self.setup_agents()

    def setup_agents(self):
        """Initialize all agents in the system"""
        self.agents[AgentType.DOCUMENT_PROCESSOR] = DocumentProcessingAgent()
        self.agents[AgentType.RETRIEVAL] = RetrievalAgent()
        self.agents[AgentType.RESPONSE_GENERATOR] = ResponseGenerationAgent()
        self.agents[AgentType.EVALUATOR] = MultiJudgeEvaluationAgent()
        self.agents[AgentType.QUERY_UNDERSTANDING] = QueryUnderstandingAgent()

        self.orchestrator = OrchestrationAgent(self.agents)

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            models_info = ollama.list()
            if hasattr(models_info, 'models') and models_info.models:
                return [model.model for model in models_info.models]
            return []
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return []

    async def process_query(self, query: str, file_upload=None) -> Dict[str, Any]:
        """Main entry point for processing queries"""
        return await self.orchestrator.process_user_query(query, file_upload)


# Streamlit UI Implementation with PDF Viewer
def main():
    st.title("🤖 DeepLaw - Horizontal Multi-Agent System")

    # Initialize system
    if "multi_agent_system" not in st.session_state:
        st.session_state.multi_agent_system = HorizontalMultiAgentSystem()
        st.session_state.conversation_history = []
        st.session_state.current_pdf_pages = []

    system = st.session_state.multi_agent_system

    # Layout with PDF viewer
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📄 PDF Viewer")

        # File upload
        file_upload = st.file_uploader("Upload PDF Document", type="pdf")

        # Display PDF pages if available
        if st.session_state.current_pdf_pages:
            st.subheader("Document Pages")
            zoom_level = st.slider("Zoom Level", 100, 300, 150)

            for i, page_img in enumerate(st.session_state.current_pdf_pages):
                if page_img is not None:
                    try:
                        # Convert pdfplumber page to displayable image
                        with st.expander(f"Page {i + 1}"):
                            st.image(page_img.original, width=zoom_level)
                    except Exception as e:
                        st.write(f"Page {i + 1}: [Image not available]")
                else:
                    st.write(f"Page {i + 1}: [Could not display]")

        # Agent status
        st.header("⚡ Agent Status")
        for agent_type, agent in system.agents.items():
            st.write(f"**{agent.name}** - 🟢 Active")

    with col2:
        st.header("💬 Multi-Agent Chat")

        # Model selection
        available_models = system.get_available_models()
        if available_models:
            selected_model = st.selectbox("Select Response Model", available_models)
        else:
            st.error("No Ollama models found. Please install models using 'ollama pull <model_name>'")
            selected_model = None

        # Display conversation history
        chat_container = st.container(height=400)
        with chat_container:
            for msg in st.session_state.conversation_history:
                if msg["role"] == "user":
                    st.chat_message("user", avatar="😎").markdown(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(msg["content"])
                        if "evaluations" in msg:
                            avg_score = sum(eval_obj["overall_score"] for eval_obj in msg["evaluations"]) / len(
                                msg["evaluations"])
                            st.caption(f"📊 Avg Evaluation: {avg_score:.1f}/10.0 ({len(msg['evaluations'])} judges)")

        # Chat input
        if prompt := st.chat_input("Ask your legal question..."):
            # Add user message
            st.session_state.conversation_history.append({"role": "user", "content": prompt})

            # Process through multi-agent system
            with st.spinner("Multi-agent processing..."):
                try:
                    result = asyncio.run(system.process_query(prompt, file_upload))

                    if "error" not in result:
                        # Update PDF pages if new document was processed
                        if "pdf_pages" in result and result["pdf_pages"]:
                            st.session_state.current_pdf_pages = result["pdf_pages"]

                        # Add assistant response
                        assistant_msg = {
                            "role": "assistant",
                            "content": result["response"],
                            "timestamp": result["timestamp"]
                        }

                        if "evaluations" in result and result["evaluations"]:
                            assistant_msg["evaluations"] = result["evaluations"]

                        st.session_state.conversation_history.append(assistant_msg)

                        # Show processing details
                        with st.expander("🔍 Processing Details"):
                            if "query_analysis" in result:
                                st.subheader("Query Analysis")
                                st.json(result["query_analysis"])

                            if "evaluations" in result:
                                st.subheader("Multi-Judge Evaluation")
                                for eval_obj in result["evaluations"]:
                                    st.write(f"**{eval_obj['judge_model']}**: {eval_obj['overall_score']:.1f}/10.0")
                                    st.caption(f"Notes: {eval_obj['evaluation_notes']}")

                            st.write(f"**Processing Time:** {result['processing_time']:.2f}s")

                except Exception as e:
                    st.error(f"Multi-agent system error: {e}")


if __name__ == "__main__":
    main()