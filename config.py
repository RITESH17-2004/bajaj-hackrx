import os
from typing import Optional

class Config:
    MISTRAL_API_KEY: Optional[str] = os.getenv('MISTRAL_API_KEY')
    MODEL_CACHE_DIR: str = os.getenv('MODEL_CACHE_DIR', '/tmp/models')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')  # Show timing information
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '2'))  # Optimize for Railway
    
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SIMILARITY_THRESHOLD: float = 0.25
    MAX_RELEVANT_CHUNKS: int = 12
    
    LLM_MODEL: str = "mistral-medium"
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 100