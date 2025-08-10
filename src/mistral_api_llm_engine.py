import os
import logging
import asyncio
import functools
import sys
from concurrent.futures import ThreadPoolExecutor
from config import Config # Import Config

try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError as e:
    logging.error("Failed to import mistralai. This is unexpected.")
    MISTRAL_AVAILABLE = False

class MistralApiLLMEngine:
    """
    A specific LLM engine designed to interact with the Mistral AI API.
    It handles API key validation, text generation, and error logging.
    """
    def __init__(self, executor: ThreadPoolExecutor = None):
        """
        Initializes the MistralApiLLMEngine.
        Raises ImportError if Mistral SDK is not installed or ValueError if API key is missing/invalid.
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError("Mistral SDK not installed. Please run `pip install mistralai`.")
        
        self.api_key = os.getenv('MISTRAL_API_KEY')
        if not self.api_key or self.api_key == 'your-mistral-api-key-here':
            raise ValueError("MISTRAL_API_KEY environment variable not set or is a placeholder.")
        
        self.client = Mistral(api_key=self.api_key)
        self.executor = executor or ThreadPoolExecutor() # Use provided executor or create a new one
        logging.info("Mistral API LLM Engine initialized for the agent.")

    async def generate(self, prompt: str) -> str:
        """
        Generates a text response from the Mistral API based on the given prompt.
        Runs the synchronous API call in a thread pool executor to prevent blocking.
        """
        loop = asyncio.get_event_loop()
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Run the synchronous Mistral API call in the executor
            response = await loop.run_in_executor(
                self.executor,
                functools.partial(
                    self.client.chat.complete,
                    model=Config.MISTRAL_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=50,
                )
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error generating text with Mistral: {e}")
            return ""