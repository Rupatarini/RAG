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

from ..config import VECTOR_STORE_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME

# -------------------------------
# Global LlamaIndex config
# -------------------------------
def configure_llama_index_settings():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY missing")

    Settings.llm = GoogleGenAI(
        model=LLM_MODEL_NAME,
        api_key=api_key
    )

    Settings.embed_model = GoogleGenAIEmbedding(
        model=EMBEDDING_MODEL_NAME,
        api_key=api_key
    )

configure_llama_index_settings()

# -------------------------------
# Helpers
# -------------------------------
def _session_path(session_id: str) -> str:
    return os.path.join(VECTOR_STORE_PATH, session_id)

def _index_exists(path: str) -> bool:
    return (
        os.path.exists(path) and
        os.path.isfile(os.path.join(path, "docstore.json"))
    )

# -------------------------------
# Index loader / creator
# -------------------------------
def get_index(session_id: str) -> VectorStoreIndex:
    session_dir = _session_path(session_id)
    os.makedirs(session_dir, exist_ok=True)

    # ✅ Load existing index
    if _index_exists(session_dir):
        storage_context = StorageContext.from_defaults(
            persist_dir=session_dir
        )
        return load_index_from_storage(storage_context)

    # ✅ Create new empty index
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex([], storage_context=storage_context)
    index.storage_context.persist(persist_dir=session_dir)
    return index

# -------------------------------
# Add documents
# -------------------------------
def add_documents(filepath: str, session_id: str) -> int:
    documents = SimpleDirectoryReader(
        input_files=[filepath]
    ).load_data()

    index = get_index(session_id)

    for doc in documents:
        index.insert(doc)

    index.storage_context.persist(
        persist_dir=_session_path(session_id)
    )

    return len(documents)

# -------------------------------
# Query index
# -------------------------------
def query_index(question: str, session_id: str):
    session_dir = _session_path(session_id)

    # 🔒 Ask before upload protection
    if not _index_exists(session_dir):
        return (
            "Please upload at least one document before asking questions.",
            []
        )

    index = get_index(session_id)
    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "Unknown")
            })

    return str(response), sources
