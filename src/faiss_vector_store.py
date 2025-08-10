import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
import logging

class FAISSVectorStore:
    """
    Manages a FAISS index for efficient storage and retrieval of document embeddings.
    Handles adding documents, searching for relevant chunks, and saving/loading the index.
    """
    def __init__(self, dimension: int = 384, executor: ThreadPoolExecutor = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension) # Initialize FAISS index with Inner Product (IP) metric
        self.documents = [] # Stores the actual document chunks (metadata)
        self.embeddings = None # Stores the embeddings corresponding to the documents
        self.executor = executor # Thread pool executor for running blocking FAISS operations asynchronously
        
    async def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Adds a batch of documents and their corresponding embeddings to the vector store.
        Runs the synchronous FAISS add operation in a thread pool.
        """
        logging.info(f"Adding {len(documents)} documents to vector store.")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            functools.partial(self._add_documents_sync, documents, embeddings)
        )
    
    def _add_documents_sync(self, documents: List[Dict], embeddings: np.ndarray):
        """
        Synchronous method to add documents and embeddings to the FAISS index.
        Normalizes embeddings before adding them.
        """
        self.documents = documents
        self.embeddings = embeddings
        
        normalized_embeddings = self._normalize_embeddings(embeddings)
        self.index.add(normalized_embeddings.astype('float32'))
        logging.info(f"Added {len(documents)} documents to FAISS index.")

    async def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Searches the vector store for the top-k most similar documents to the query embedding.
        Runs the synchronous FAISS search operation in a thread pool.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(self._search_sync, query_embedding, k)
        )

    def _search_sync(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Synchronous method to perform the FAISS search.
        """
        if self.index.ntotal == 0:
            return []
        
        normalized_query = self._normalize_embeddings(query_embedding.reshape(1, -1))
        
        # Perform the search using FAISS
        scores, indices = self.index.search(normalized_query.astype('float32'), min(k, self.index.ntotal))
        
        results = []
        # Map the results back to the original documents and their scores
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    async def search_with_threshold(self, query_embedding: np.ndarray, threshold: float = 0.7, k: int = 10) -> List[Tuple[Dict, float]]:
        """
        Searches the vector store and returns documents with similarity scores above a given threshold.
        """
        results = await self.search(query_embedding, k)
        return [(doc, score) for doc, score in results if score >= threshold]
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Normalizes embeddings to unit vectors. Essential for Inner Product (IP) similarity.
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1 # Avoid division by zero for zero vectors
        return embeddings / norms
    
    async def save_index(self, filepath: str, full_document_text: str):
        """
        Saves the FAISS index and associated metadata (documents, embeddings, full document text) to disk.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            functools.partial(self._save_index_sync, filepath, full_document_text)
        )

    def _save_index_sync(self, filepath: str, full_document_text: str):
        """
        Synchronous method to save the FAISS index and metadata.
        """
        faiss.write_index(self.index, f"{filepath}.faiss")
        self._save_metadata(filepath, full_document_text)
        logging.info(f"Vector store index saved to {filepath}.faiss and metadata to {filepath}_metadata.pkl.")

    def _save_metadata(self, filepath: str, full_document_text: str):
        """
        Saves the metadata (documents, embeddings, dimension, full document text) using pickle.
        """
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'dimension': self.dimension,
                'full_document_text': full_document_text
            }, f)
    
    async def load_index(self, filepath: str) -> Tuple[bool, str]:
        """
        Loads the FAISS index and associated metadata from disk.
        Returns True and the full document text if successful, False and empty string otherwise.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(self._load_index_sync, filepath)
        )

    def _load_index_sync(self, filepath: str) -> Tuple[bool, str]:
        """
        Synchronous method to load the FAISS index and metadata.
        """
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}_metadata.pkl"):
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            metadata = self._load_metadata(filepath)
            self.documents = metadata['documents']
            self.embeddings = metadata['embeddings']
            self.dimension = metadata['dimension']
            full_document_text = metadata.get('full_document_text', '') # Retrieve full document text
            logging.info(f"Vector store index and metadata loaded from {filepath}.faiss and {filepath}_metadata.pkl.")
            return True, full_document_text
        logging.info(f"No existing vector store found at {filepath}.faiss or {filepath}_metadata.pkl.")
        return False, ""

    def _load_metadata(self, filepath: str):
        """
        Loads the metadata from the pickle file.
        """
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            return pickle.load(f)
    
    async def get_document_count(self) -> int:
        """
        Returns the number of documents currently stored in the vector store.
        """
        return len(self.documents)
    
    async def clear(self):
        """
        Clears the FAISS index and resets stored documents and embeddings.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.index.reset # Resets the FAISS index
        )
        self.documents = []
        self.embeddings = None
        logging.info("Vector store cleared.")
