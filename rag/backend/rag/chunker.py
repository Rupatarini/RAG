import os
from typing import List # <-- CRITICAL FIX: Import List for type hinting

from llama_index.core.node_parser import SentenceSplitter 
from llama_index.core.schema import Document 

# Configuration for chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20

# ----------------------------------------------------------------------
# 1. Document Loading/Wrapper (The problematic function)
# ----------------------------------------------------------------------

# The function signature will now correctly recognize 'List'
def get_text_from_file(filepath: str) -> List[Document]:
    """
    Loads text content from a file (PDF or TXT) and wraps it in a 
    LlamaIndex Document object.
    """
    filename = os.path.basename(filepath)

    text = ""
    if filepath.lower().endswith('.pdf'):
        try:
            from .pdf_utils import extract_text_from_pdf
            text = extract_text_from_pdf(filepath)
        except ImportError:
            print("Warning: pdf_utils.py not found. Reading PDF as plain text.")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    else: # Handles .txt and other plain text files
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

    # Create a base Document object
    document = Document(
        text=text, 
        metadata={"filename": filename}
    )
    
    return [document]


# ----------------------------------------------------------------------
# 2. Document Chunking/Splitting
# ----------------------------------------------------------------------

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of Document objects into smaller chunks (Nodes).
    """
    
    # Initialize the text splitter
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    
    # Split the documents into nodes (chunks)
    nodes = splitter.get_nodes_from_documents(documents)
    
    return nodes