"""
Embedding Store - Manages persistent storage of document embeddings using FAISS.
Ensures embeddings are not regenerated unnecessarily by caching them to disk.
"""

import os
import pickle
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class EmbeddingStoreManager:
    """
    Manages persistent embedding storage.
    Stores FAISS index and document metadata to disk so embeddings
    are generated only once per document.
    """

    def __init__(
        self,
        store_dir: str = "vector_store",
        embedding_model: str = "text-embedding-ada-002",
    ):
        """
        Initialize the embedding store manager.

        Args:
            store_dir: Directory to store FAISS indexes and metadata
            embedding_model: OpenAI embedding model name
        """
        self.store_dir = store_dir
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store: Optional[FAISS] = None
        self.documents_metadata: Dict[str, Dict[str, Any]] = {}
        self._ensure_store_dir()
        self._load_metadata()

    def _ensure_store_dir(self):
        """Create store directory if it doesn't exist."""
        for subdir in ["", "indexes", "chunks"]:
            path = os.path.join(self.store_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    def _get_file_hash(self, file_path: str) -> str:
        """Generate a hash for a file to track if it has been modified."""
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()

    def _get_document_id(self, file_name: str, file_hash: str) -> str:
        """Generate a unique document ID."""
        return f"{file_name}_{file_hash[:12]}"

    def _load_metadata(self):
        """Load metadata from disk."""
        metadata_path = os.path.join(self.store_dir, "metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    self.documents_metadata = json.load(f)
            except Exception:
                self.documents_metadata = {}

    def _save_metadata(self):
        """Save metadata to disk."""
        metadata_path = os.path.join(self.store_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(self.documents_metadata, f, indent=2)

    def is_document_processed(self, file_path: str) -> bool:
        """Check if a document has already been processed (embedding exists)."""
        if not os.path.exists(file_path):
            return False
        file_hash = self._get_file_hash(file_path)
        file_name = os.path.basename(file_path)
        doc_id = self._get_document_id(file_name, file_hash)
        return doc_id in self.documents_metadata

    def get_document_ids(self) -> List[str]:
        """Get list of all processed document IDs."""
        return list(self.documents_metadata.keys())

    def get_document_summaries(self) -> List[Dict[str, Any]]:
        """Get summaries of all processed documents for UI display."""
        summaries = []
        for doc_id, meta in self.documents_metadata.items():
            summaries.append(
                {
                    "doc_id": doc_id,
                    "file_name": meta.get("file_name", "Unknown"),
                    "file_type": meta.get("file_type", "Unknown"),
                    "chunks": meta.get("chunk_count", 0),
                    "processed_at": meta.get("processed_at", "Unknown"),
                    "file_size_kb": meta.get("file_size", 0) / 1024,
                }
            )
        return summaries

    def process_and_store_document(
        self, file_path: str, file_name: str, chunks: List[Document]
    ) -> Dict[str, Any]:
        """
        Process document chunks into embeddings and store persistently.

        Args:
            file_path: Path to the original file
            file_name: Name of the file
            chunks: List of document chunks

        Returns:
            Metadata about the processed document
        """
        file_hash = self._get_file_hash(file_path)
        doc_id = self._get_document_id(file_name, file_hash)

        if doc_id in self.documents_metadata:
            return self.documents_metadata[doc_id]

        # Save chunks to disk for potential reprocessing
        chunks_path = os.path.join(self.store_dir, "chunks", f"{doc_id}.pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)

        # Create FAISS vector store from chunks
        vector_store = FAISS.from_documents(chunks, self.embeddings)

        # Save FAISS index to disk
        index_path = os.path.join(self.store_dir, "indexes", doc_id)
        vector_store.save_local(index_path)

        # If there's an existing main store, merge with it
        main_index_path = os.path.join(self.store_dir, "main_index")
        if os.path.exists(os.path.join(main_index_path, "index.faiss")):
            try:
                existing_store = FAISS.load_local(
                    main_index_path, self.embeddings, allow_dangerous_deserialization=True
                )
                existing_store.merge_from(vector_store)
                existing_store.save_local(main_index_path)
            except Exception:
                # If merge fails, create new main index
                vector_store.save_local(main_index_path)
        else:
            vector_store.save_local(main_index_path)

        # Update metadata
        file_size = os.path.getsize(file_path)
        self.documents_metadata[doc_id] = {
            "file_name": file_name,
            "file_path": file_path,
            "file_hash": file_hash,
            "chunk_count": len(chunks),
            "file_size": file_size,
            "processed_at": datetime.now().isoformat(),
        }
        self._save_metadata()

        return self.documents_metadata[doc_id]

    def load_vector_store(self) -> Optional[FAISS]:
        """
        Load the main vector store from disk.
        Returns None if no index exists.
        """
        main_index_path = os.path.join(self.store_dir, "main_index")
        if os.path.exists(os.path.join(main_index_path, "index.faiss")):
            try:
                self.vector_store = FAISS.load_local(
                    main_index_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
                return self.vector_store
            except Exception as e:
                print(f"Error loading vector store: {e}")
                return None
        return None

    def get_vector_store(self) -> Optional[FAISS]:
        """Get the current vector store, loading from disk if needed."""
        if self.vector_store is None:
            return self.load_vector_store()
        return self.vector_store

    def similarity_search(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents in the vector store.

        Args:
            query: The user's question
            k: Number of results to return

        Returns:
            List of (Document, score) tuples
        """
        vector_store = self.get_vector_store()
        if vector_store is None:
            return []

        results = vector_store.similarity_search_with_score(query, k=k)
        return results

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document and its embeddings from the store.
        Note: FAISS doesn't support deletion easily, so we rebuild.
        """
        if doc_id not in self.documents_metadata:
            return False

        # Remove chunks file
        chunks_path = os.path.join(self.store_dir, "chunks", f"{doc_id}.pkl")
        if os.path.exists(chunks_path):
            os.remove(chunks_path)

        # Remove individual index
        index_path = os.path.join(self.store_dir, "indexes", doc_id)
        if os.path.exists(index_path):
            import shutil
            shutil.rmtree(index_path)

        # Remove from metadata
        del self.documents_metadata[doc_id]
        self._save_metadata()

        # Rebuild main index from remaining documents
        self._rebuild_main_index()

        return True

    def _rebuild_main_index(self):
        """Rebuild the main FAISS index from all remaining document chunks."""
        main_index_path = os.path.join(self.store_dir, "main_index")

        all_chunks = []
        for doc_id in self.documents_metadata:
            chunks_path = os.path.join(self.store_dir, "chunks", f"{doc_id}.pkl")
            if os.path.exists(chunks_path):
                with open(chunks_path, "rb") as f:
                    chunks = pickle.load(f)
                    all_chunks.extend(chunks)

        if all_chunks:
            vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            vector_store.save_local(main_index_path)
        else:
            # Remove main index if no documents left
            if os.path.exists(main_index_path):
                import shutil
                shutil.rmtree(main_index_path)

            self.vector_store = None

    def clear_all(self):
        """Clear all stored data."""
        import shutil

        if os.path.exists(self.store_dir):
            shutil.rmtree(self.store_dir)
        self._ensure_store_dir()
        self.documents_metadata = {}
        self.vector_store = None
        self._save_metadata()