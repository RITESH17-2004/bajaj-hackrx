from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import tiktoken

class EmbeddingEngine:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 512
    
    def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        texts = []
        for chunk in chunks:
            text = chunk['text']
            if self._count_tokens(text) > self.max_tokens:
                text = self._truncate_text(text)
            texts.append(text)
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        if self._count_tokens(query) > self.max_tokens:
            query = self._truncate_text(query)
        
        embedding = self.model.encode([query], convert_to_numpy=True)
        return embedding[0]
    
    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))
    
    def _truncate_text(self, text: str) -> str:
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= self.max_tokens:
            return text
        
        truncated_tokens = tokens[:self.max_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()