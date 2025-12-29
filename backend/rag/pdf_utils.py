import fitz  # PyMuPDF, typically installed with pypdf dependencies or explicitly.
import os

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text content from a PDF file.

    Note: For simpler RAG setups using LlamaIndex/PyPDFLoader, 
    this manual extraction might be optional, but it's good practice 
    for robust, custom processing.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        raise
        
    return text

def extract_metadata_from_pdf(pdf_path: str) -> dict:
    """
    Extracts metadata like title, author, and creation date from a PDF.
    """
    metadata = {"filename": os.path.basename(pdf_path)}
    try:
        with fitz.open(pdf_path) as doc:
            info = doc.metadata
            metadata["title"] = info.get("title")
            metadata["author"] = info.get("author")
            metadata["creation_date"] = info.get("creationDate")
    except Exception:
        # Ignore errors if metadata can't be read
        pass
        
    return metadata