import os
import shutil
from typing import Tuple, List

from llama_index.core import (
    VectorStoreIndex,
    Settings,
)
from llama_index.core.schema import NodeWithScore

from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

from ..config import VECTOR_STORE_PATH, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME

# ============================================================
# GLOBAL SINGLETON INDEX (IMPORTANT)
# ============================================================

_INDEX: VectorStoreIndex | None = None


# ============================================================
# CLEAR VECTOR STORE ON APP START
# ============================================================

if os.path.exists(VECTOR_STORE_PATH):
    shutil.rmtree(VECTOR_STORE_PATH)

os.makedirs(VECTOR_STORE_PATH, exist_ok=True)


# ============================================================
# CONFIGURE LLAMAINDEX (ONCE)
# ============================================================

def configure_llama_index():
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


configure_llama_index()


# ============================================================
# GET OR CREATE INDEX (SINGLE INSTANCE)
# ============================================================

def get_index() -> VectorStoreIndex:
    global _INDEX

    if _INDEX is None:
        _INDEX = VectorStoreIndex([])

    return _INDEX


# ============================================================
# ADD DOCUMENTS
# ============================================================

def add_documents(filepath: str) -> int:
    from llama_index.core import SimpleDirectoryReader

    index = get_index()

    docs = SimpleDirectoryReader(
        input_files=[filepath]
    ).load_data()

    index.insert_nodes(docs)

    return len(docs)


# ============================================================
# QUERY
# ============================================================

def query_index(question: str) -> Tuple[str, List[dict]]:
    index = get_index()

    if len(index.docstore.docs) == 0:
        return "No documents indexed yet. Please upload a document first.", []

    query_engine = index.as_query_engine(similarity_top_k=5)
    response = query_engine.query(question)

    sources = []
    if hasattr(response, "source_nodes"):
        for node in response.source_nodes:
            sources.append({
                "filename": node.metadata.get("filename", "Unknown")
            })

    return str(response), sources
