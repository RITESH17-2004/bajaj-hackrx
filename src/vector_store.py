import faiss
import numpy as np
from typing import List, Dict, Tuple
import pickle
import os

class VectorStore:
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []
        self.embeddings = None
        
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        self.documents = documents
        self.embeddings = embeddings
        
        normalized_embeddings = self._normalize_embeddings(embeddings)
        self.index.add(normalized_embeddings.astype('float32'))
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        if self.index.ntotal == 0:
            return []
        
        normalized_query = self._normalize_embeddings(query_embedding.reshape(1, -1))
        
        scores, indices = self.index.search(normalized_query.astype('float32'), min(k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def search_with_threshold(self, query_embedding: np.ndarray, threshold: float = 0.7, k: int = 10) -> List[Tuple[Dict, float]]:
        results = self.search(query_embedding, k)
        return [(doc, score) for doc, score in results if score >= threshold]
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return embeddings / norms
    
    def save_index(self, filepath: str):
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'embeddings': self.embeddings,
                'dimension': self.dimension
            }, f)
    
    def load_index(self, filepath: str):
        if os.path.exists(f"{filepath}.faiss") and os.path.exists(f"{filepath}_metadata.pkl"):
            self.index = faiss.read_index(f"{filepath}.faiss")
            
            with open(f"{filepath}_metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
                self.documents = metadata['documents']
                self.embeddings = metadata['embeddings']
                self.dimension = metadata['dimension']
            
            return True
        return False
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def clear(self):
        self.index.reset()
        self.documents = []
        self.embeddings = None