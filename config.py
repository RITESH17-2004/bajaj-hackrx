import os
from typing import Optional

class Config:
    """
    Configuration class for the application, managing various settings
    such as API keys, model names, logging levels, and resource limits.
    Settings are primarily loaded from environment variables.
    """
    # API Keys and Tokens
    MISTRAL_API_KEY: Optional[str] = os.getenv('MISTRAL_API_KEY')
    BEARER_TOKEN: Optional[str] = os.getenv('BEARER_TOKEN')

    # Directory for caching models
    MODEL_CACHE_DIR: str = os.getenv('MODEL_CACHE_DIR', '/tmp/models')

    # Logging level (e.g., 'INFO', 'DEBUG', 'WARNING')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    # Maximum number of worker threads for concurrent operations
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '2'))

    # Embedding model configuration
    EMBEDDING_MODEL: str = "paraphrase-MiniLM-L3-v2"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    SIMILARITY_THRESHOLD: float = 0.25
    MAX_RELEVANT_CHUNKS: int = 12

    # Large Language Model (LLM) configuration
    LLM_MODEL: str = "mistral-medium"
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 100

    # Specific Mistral model name, configurable via environment variable
    MISTRAL_MODEL_NAME: str = os.getenv('MISTRAL_MODEL_NAME', 'mistral-small-latest')
