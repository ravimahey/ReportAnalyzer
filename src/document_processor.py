"""
Document Processor - Handles loading and parsing various document types.
Supports: PDF, CSV, Excel, DOCX, TXT
"""

import os
import tempfile
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {
    ".pdf": "PDF Document",
    ".csv": "CSV File",
    ".xlsx": "Excel File",
    ".xls": "Excel File",
    ".docx": "Word Document",
    ".doc": "Word Document",
    ".txt": "Text File",
}


def get_loader_for_file(file_path: str):
    """Returns the appropriate document loader based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext == ".csv":
        return CSVLoader(file_path)
    elif ext in [".xlsx", ".xls"]:
        return UnstructuredExcelLoader(file_path, mode="elements")
    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path)
    elif ext == ".txt":
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def load_document(file_path: str) -> List[Document]:
    """
    Load a document from a file path.
    Returns a list of LangChain Document objects.
    """
    loader = get_loader_for_file(file_path)
    documents = loader.load()
    return documents


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def process_uploaded_file(
    uploaded_file, upload_dir: str = "uploaded_docs"
) -> Dict[str, Any]:
    """
    Process an uploaded file from Streamlit.
    Saves the file locally and returns metadata.
    """
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    ext = os.path.splitext(file_path)[1].lower()
    file_size = os.path.getsize(file_path)

    return {
        "file_name": uploaded_file.name,
        "file_path": file_path,
        "file_type": SUPPORTED_EXTENSIONS.get(ext, "Unknown"),
        "file_size": file_size,
        "file_extension": ext,
    }