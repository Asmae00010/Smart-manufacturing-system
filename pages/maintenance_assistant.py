import streamlit as st
import os
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
DOCUMENTS_DIR = "documents"
MAINTENANCE_LOGS_FILE = "enhanced_maintenance_logs.csv"
DB_DIR = "vector_db"
OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual API key

# Custom prompt template
CUSTOM_PROMPT = PromptTemplate.from_template("""You are a friendly and knowledgeable maintenance assistant for a steel manufacturing facility. Your name is Steel Assistant. Adapt your response style based on the type of question:

For casual questions (greetings, general inquiries about your capabilities, or non-technical questions):
- Respond naturally and conversationally
- Keep responses brief but friendly
- Skip formal formatting

For technical or maintenance-related questions, provide detailed responses with:
1. Brief context from available data
2. Clear, numbered main points
3. Specific details including:
   - Numbers and measurements
   - Technical specifications
   - Timing information
4. Relevant recommendations

Context: {context}
Chat History: {chat_history}
Question: {question}

Response:""")

class CustomDirectoryLoader:
    def __init__(self, directory):
        self.directory = directory
    
    def load(self):
        documents = []
        for file_path in Path(self.directory).rglob("*"):
            try:
                if file_path.suffix.lower() in ['.pdf', '.txt', '.md']:
                    logger.info(f"Loading {file_path.suffix} file: {file_path}")
                    if file_path.suffix.lower() == '.pdf':
                        loader = PyPDFLoader(str(file_path))
                    else:
                        loader = TextLoader(str(file_path))
                    documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading file {file_path}: {str(e)}")
        return documents

class CombinedMaintenanceRAG:
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.conversation_chain = None
        self.initialize_system()

    def initialize_embeddings(self):
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    def load_maintenance_logs(self):
        if not os.path.exists(MAINTENANCE_LOGS_FILE):
            logger.warning(f"{MAINTENANCE_LOGS_FILE} not found.")
            return []
        
        df = pd.read_csv(MAINTENANCE_LOGS_FILE)
        return [f"{row.to_dict()}" for _, row in df.iterrows()]

    def initialize_system(self):
        if os.path.exists(DB_DIR):
            # Load existing vector store
            self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            self.vector_store = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        else:
            # Create new vector store
            self.initialize_embeddings()
            self.ingest_all_documents()
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY),
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
        )

    def ingest_all_documents(self):
        all_texts = []
        
        maintenance_logs = self.load_maintenance_logs()
        if maintenance_logs:
            all_texts.extend(maintenance_logs)
        
        if os.path.exists(DOCUMENTS_DIR):
            loader = CustomDirectoryLoader(DOCUMENTS_DIR)
            documents = loader.load()
            if documents:
                all_texts.extend([doc.page_content for doc in documents])
        
        if not all_texts:
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_text("\n".join(all_texts))
        
        self.vector_store = Chroma.from_texts(
            texts=splits,
            embedding=self.embeddings,
            persist_directory=DB_DIR
        )
        self.vector_store.persist()

    def get_response(self, query: str) -> str:
        try:
            return self.conversation_chain.run(query)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

def show_page():
    """Display the maintenance assistant page"""
    st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #f0f2f6;
        }
        .chat-message.assistant {
            background-color: #e6f3ff;
        }
        .chat-timestamp {
            color: #666;
            font-size: 0.8rem;
            margin-top: 0.5rem;
        }
        .main-header {
            color: #1F618D;
            padding: 1rem;
            background-color: #f8f9fa;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-header">Maintenance Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize chat messages
    if 'maintenance_messages' not in st.session_state:
        st.session_state.maintenance_messages = []
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.maintenance_messages.append({
            "role": "assistant",
            "content": "Welcome to the Steel Manufacturing Maintenance Assistant. How can I help you today?",
            "timestamp": current_time
        })

    # Initialize RAG system
    if 'maintenance_rag_system' not in st.session_state:
        try:
            with st.spinner("Initializing maintenance system..."):
                st.session_state.maintenance_rag_system = CombinedMaintenanceRAG()
            st.success("✅ Maintenance system initialized!")
        except Exception as e:
            st.error(f"Error initializing maintenance system: {str(e)}")
            return

    # Display chat messages
    for message in st.session_state.maintenance_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timestamp" in message:
                st.caption(message["timestamp"])

    # Chat input
    if prompt := st.chat_input("Ask about maintenance..."):
        # Add user message
        current_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.maintenance_messages.append({
            "role": "user",
            "content": prompt,
            "timestamp": current_time
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(current_time)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            try:
                with st.spinner("Processing your question..."):
                    response = st.session_state.maintenance_rag_system.get_response(prompt)
                    st.markdown(response)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    st.caption(current_time)
                    st.session_state.maintenance_messages.append({
                        "role": "assistant",
                        "content": response,
                        "timestamp": current_time
                    })
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Maintenance Assistant",
        page_icon="🛠️",
        layout="wide"
    )
    show_page()