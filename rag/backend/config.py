import os
from dotenv import load_dotenv

# --- Load Environment Variables from .env ---
load_dotenv() 
# ---------------------------------------------

# Base directory for the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory where uploaded files will be saved
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

# Configuration from .env or fallback
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "gemini-embedding-001")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash") 
VECTOR_STORE_PATH = os.path.join(BASE_DIR, 'storage')