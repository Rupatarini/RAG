import os

# -------------------------------
# Base directory
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------
# Uploads
# -------------------------------
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Vector store persistence
# -------------------------------
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "storage")
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# -------------------------------
# Model configuration
# -------------------------------
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "models/embedding-001"
)

LLM_MODEL_NAME = os.getenv(
    "LLM_MODEL_NAME",
    "models/gemini-2.5-flash"
)
