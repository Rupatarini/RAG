import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    load_index_from_storage,
)
from llama_index.core.storage.storage_context import StorageContext

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


from .chunker import get_text_from_file
from ..config import VECTOR_STORE_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME

# -------------------------------
# GLOBAL CONFIG (RUN ON STARTUP)
# -------------------------------

def configure_llama_index_settings():
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing")

    # Gemini LLM
    Settings.llm = GoogleGenAI(
        model=LLM_MODEL_NAME,
        api_key=os.getenv("GEMINI_API_KEY")
    )

    # Gemini Embeddings
    Settings.embed_model = GoogleEmbedding(
        model=EMBEDDING_MODEL_NAME,
        api_key=os.getenv("GEMINI_API_KEY")
    )

configure_llama_index_settings()

# -------------------------------
# INDEX MANAGEMENT
# -------------------------------

def get_index():
    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    if not os.listdir(VECTOR_STORE_PATH):
        return VectorStoreIndex(
            [],
            storage_context=StorageContext.from_defaults()
        )

    try:
        storage_context = StorageContext.from_defaults(
            persist_dir=VECTOR_STORE_PATH
        )
        return load_index_from_storage(storage_context)

    except Exception as e:
        print(f"Index load failed: {e}")
        return VectorStoreIndex(
            [],
            storage_context=StorageContext.from_defaults()
        )

# -------------------------------
# ADD DOCUMENTS
# -------------------------------

def add_documents(filepath):
    documents = SimpleDirectoryReader(
        input_files=[filepath]
    ).load_data()

    index = get_index()

    for doc in documents:
        index.insert(doc)

    index.storage_context.persist(persist_dir=VECTOR_STORE_PATH)
    return len(documents)

# -------------------------------
# QUERY INDEX
# -------------------------------

def query_index(question):
    index = get_index()

    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "Unknown")
            })

    return str(response), sources
