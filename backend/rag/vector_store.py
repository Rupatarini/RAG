import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.storage.storage_context import StorageContext
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from ..config import VECTOR_STORE_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME


# -------------------------------
# GLOBAL LLM CONFIG (ONCE)
# -------------------------------
def configure():
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

configure()


# -------------------------------
# SESSION INDEX HANDLING
# -------------------------------
def get_index(session_id: str):
    session_path = os.path.join(VECTOR_STORE_PATH, session_id)
    os.makedirs(session_path, exist_ok=True)

    if os.listdir(session_path):
        storage_context = StorageContext.from_defaults(
            persist_dir=session_path
        )
        return VectorStoreIndex.load_from_storage(storage_context)

    return VectorStoreIndex(
        [],
        storage_context=StorageContext.from_defaults(
            persist_dir=session_path
        )
    )


# -------------------------------
# ADD DOCUMENTS
# -------------------------------
def add_documents(filepath: str, session_id: str):
    docs = SimpleDirectoryReader(
        input_files=[filepath]
    ).load_data()

    index = get_index(session_id)

    for doc in docs:
        index.insert(doc)

    index.storage_context.persist()
    return len(docs)


# -------------------------------
# QUERY
# -------------------------------
def query_index(question: str, session_id: str):
    index = get_index(session_id)

    if not index.docstore.docs:
        return "No documents uploaded yet.", []

    query_engine = index.as_query_engine()
    response = query_engine.query(question)

    sources = []
    for node in getattr(response, "source_nodes", []):
        sources.append({
            "metadata": {
                "filename": node.metadata.get("filename", "Unknown")
            }
        })

    return str(response), sources
