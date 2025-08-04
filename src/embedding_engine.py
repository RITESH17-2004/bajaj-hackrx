from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict
import tiktoken
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
import logging

class EmbeddingEngine:
    def __init__(self, model_name: str = "paraphrase-MiniLM-L3-v2", device: str = "cpu", executor: ThreadPoolExecutor = None):
        self.model = SentenceTransformer(model_name).to(device)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = 512
        self.executor = executor
    
    async def generate_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        logging.info(f"[EmbeddingEngine] Generating embeddings for {len(chunks)} chunks...")
        texts = []
        for chunk in chunks:
            text = chunk['text']
            if self._count_tokens(text) > self.max_tokens:
                text = self._truncate_text(text)
            texts.append(text)
        
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            functools.partial(self.model.encode, texts, convert_to_numpy=True)
        )
        logging.info("[EmbeddingEngine] Embeddings generated successfully.")
        return embeddings
    
    async def generate_query_embedding(self, query: str) -> np.ndarray:
        logging.info(f"[EmbeddingEngine] Generating query embedding for: {query[:50]}...")
        if self._count_tokens(query) > self.max_tokens:
            query = self._truncate_text(query)
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            self.executor,
            functools.partial(self.model.encode, [query], convert_to_numpy=True)
        )
        logging.info("[EmbeddingEngine] Query embedding generated successfully.")
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