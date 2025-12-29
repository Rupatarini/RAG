import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.readers.base import Document
from llama_index.core import load_index_from_storage 

# --- Correct Google Integrations ---
from llama_index.llms.google_genai import GoogleGenAI as Gemini 
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding 

from .chunker import get_text_from_file # Correct relative import for sibling module
from ..config import VECTOR_STORE_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME # Correct relative import for config sibling

# --- Global Initialization (Using Settings) ---

def configure_llama_index_settings():
    """Initializes LlamaIndex global Settings with Gemini LLM and Embeddings."""
    
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY not found in environment variables. Ensure .env is in project root and run with $env:PYTHONPATH='..'.")

    # 1. Initialize the LLM (for response generation)
    llm = Gemini(model=LLM_MODEL_NAME)
    
    # 2. Initialize the Embedding Model (for vector creation/retrieval)
    embed_model = GoogleGenAIEmbedding(model_name=EMBEDDING_MODEL_NAME) 
    
    # 3. Apply the settings globally
    Settings.llm = llm # Assign the Gemini LLM
    Settings.embed_model = embed_model # Assign the Google Embedding Model

# Run configuration once at startup
configure_llama_index_settings()

# --- Index Management ---

def get_index():
    """Load or create the VectorStoreIndex. Uses global Settings automatically."""
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
    
    # Check if storage exists
    if not os.listdir(VECTOR_STORE_PATH):
        storage_context = StorageContext.from_defaults()
        return VectorStoreIndex([], storage_context=storage_context)
    else:
        try:
            storage_context = StorageContext.from_defaults(persist_dir=VECTOR_STORE_PATH)
            return load_index_from_storage(storage_context=storage_context)
        except Exception as e:
            print(f"Failed to load index: {e}") 
            return VectorStoreIndex([], storage_context=StorageContext.from_defaults())

# --- Document Handling ---

def add_documents(filepath):
    """Load, chunk, and index a document."""
    
    # Uses SimpleDirectoryReader for all file types
    documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
    
    index = get_index()
    
    for doc in documents:
        index.insert(doc) # Uses Settings.embed_model implicitly

    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH)
    
    return len(documents) 

# --- Query Handling ---

def query_index(question):
    """Query the RAG index and return answer and sources."""
    
    index = get_index()
    
    query_engine = index.as_query_engine() # Uses Settings.llm implicitly
    
    response = query_engine.query(question)
    
    sources = []
    for node in response.source_nodes:
        sources.append({
            'metadata': {
                'filename': node.metadata.get('filename', 'Unknown Document'),
                'chunk_index': node.node_id.split('-')[-1] 
            }
        })

    # CRITICAL FIX: Return both the answer string and the sources list
    return str(response), sources