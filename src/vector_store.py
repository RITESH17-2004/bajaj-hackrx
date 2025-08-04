import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
import logging

class VectorStore:
    def __init__(self, dimension: int = 384, executor: ThreadPoolExecutor = None):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
        self.embeddings = None
        self.executor = executor
        
    async def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        logging.info(f"[VectorStore] Adding {len(documents)} documents to vector store...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            functools.partial(self._add_documents_sync, documents, embeddings)
        )
    
    def _add_documents_sync(self, documents: List[Dict], embeddings: np.ndarray):
        self.documents = documents
        self.embeddings = embeddings
        
        normalized_embeddings = self._normalize_embeddings(embeddings)
        self.index.add(normalized_embeddings.astype('float32'))
        logging.info(f"[VectorStore] Added {len(documents)} documents.")

    async def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(self._search_sync, query_embedding, k)
        )

    def _search_sync(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        
        normalized_query = self._normalize_embeddings(query_embedding.reshape(1, -1))
        
        scores, indices = self.index.search(normalized_query.astype('float32'), min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    async def search_with_threshold(self, query_embedding: np.ndarray, threshold: float = 0.7, k: int = 10) -> List[Tuple[Dict, float]]:
        results = await self.search(query_embedding, k)
        return [(doc, score) for doc, score in results if score >= threshold]
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    
    async def save_index(self, filepath: str):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            functools.partial(self._save_index_sync, filepath)
        )

    def _save_index_sync(self, filepath: str):
        faiss.write_index(self.index, f"{filepath}.faiss")
        self._save_metadata(filepath)

    def _save_metadata(self, filepath: str):
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'dimension': self.dimension
            }, f)
    
    async def load_index(self, filepath: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(self._load_index_sync, filepath)
        )

    def _load_index_sync(self, filepath: str):
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}_metadata.pkl"):
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            metadata = self._load_metadata(filepath)
            self.documents = metadata['documents']
            self.embeddings = metadata['embeddings']
            self.dimension = metadata['dimension']
            
            return True
        return False

    def _load_metadata(self, filepath: str):
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            return pickle.load(f)
    
    async def get_document_count(self) -> int:
        return len(self.documents)
    
    async def clear(self):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self.index.reset
        )
        self.documents = []
        self.embeddings = None