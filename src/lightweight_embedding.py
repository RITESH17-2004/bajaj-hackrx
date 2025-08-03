import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import re

class LightweightEmbedding:
    """Memory-efficient text embedding using TF-IDF"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit vocabulary
            stop_words='english',
            lowercase=True,
            strip_accents='ascii'
        )
        self.embeddings = None
        self.texts = []
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings for texts"""
        self.texts = texts
        self.embeddings = self.vectorizer.fit_transform(texts)
        return self.embeddings.toarray()
    
    def get_embedding_dimension(self) -> int:
        """Return embedding dimension"""
        return 1000  # max_features
    
    def find_similar(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Find most similar texts to query"""
        if self.embeddings is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.embeddings).flatten()
        
        # Get top-k indices and scores
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(idx, similarities[idx]) for idx in top_indices if similarities[idx] > 0.1]
        
        return results