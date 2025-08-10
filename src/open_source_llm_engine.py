from transformers import pipeline
import torch
from typing import List, Dict, Tuple
import re
import asyncio
import functools
import logging
import json
from src.text_cleaner_utils import clean_escape_characters

class OpenSourceLLMEngine:
    """
    Provides LLM capabilities using free, open-source models from Hugging Face Transformers.
    Includes pipelines for question-answering and text generation.
    """
    def __init__(self, device: str = "cpu"):
        logging.info("Loading open-source LLM models (this may take a few minutes the first time)...")

        # Initialize question-answering pipeline
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad",
            device=0 if device == "cuda" else -1  # Use GPU if available, otherwise CPU
        )

        # Initialize text generation pipeline
        self.text_generator = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-2.7B",
            truncation=True,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256,
            device=0 if device == "cuda" else -1
        )

        logging.info("Open-source LLM models loaded successfully!")

    async def generate(self, prompt: str) -> str:
        """
        Generates text based on a given prompt using the text generation pipeline.
        Runs the synchronous generation in a thread pool executor.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self._generate_sync, prompt)
        )

    def _generate_sync(self, prompt: str) -> str:
        """
        Synchronous text generation using the Hugging Face pipeline.
        """
        try:
            results = self.text_generator(prompt)
            return self._post_process_json(results[0]["generated_text"])
        except Exception as e:
            logging.error(f"Error with open-source LLM text generation: {e}")
            return ""

    def _post_process_json(self, text: str) -> str:
        """
        Extracts and validates a JSON object from the generated text.
        """
        # Find the first and last curly brace to extract the JSON object
        start_index = text.find('{')
        end_index = text.rfind('}')
        if start_index != -1 and end_index != -1:
            json_text = text[start_index:end_index+1]
            try:
                # Validate the JSON
                json.loads(json_text)
                return json_text
            except json.JSONDecodeError:
                pass
        return ""

    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        """
        Generates an answer to a query using the question-answering pipeline.
        Handles mathematical content specially and provides a fallback mechanism.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self._generate_answer_sync, query, relevant_chunks, query_intent)
        )

    def _generate_answer_sync(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        """
        Synchronous answer generation using the Hugging Face question-answering pipeline.
        """
        if not relevant_chunks:
            return "I couldn't find relevant information in the provided policy documents to answer this question."

        # Combine top relevant chunks as context for the QA model
        context = self._prepare_context(relevant_chunks[:3])
        
        # Handle mathematical content separately if detected
        content_type = query_intent.get('content_type', 'unknown')
        if content_type == 'mathematical':
            return self._handle_mathematical_content(query, context, relevant_chunks)

        try:
            # Use the question-answering pipeline to find the answer in the context
            result = self.qa_pipeline(
                question=query,
                context=context,
                max_answer_len=100
            )

            answer = result['answer']
            confidence = result['score']

            # If confidence is too low, try a rule-based fallback approach
            if confidence < 0.3:
                answer = self._generate_fallback_answer(query, relevant_chunks, query_intent)

            return self._post_process_answer(answer, relevant_chunks)

        except Exception as e:
            logging.error(f"Error with open-source LLM question-answering: {e}")
            return self._generate_fallback_answer(query, relevant_chunks, query_intent)
    
    def _handle_mathematical_content(self, query: str, context: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """
        Handles mathematical questions by extracting and reporting exact content from the source.
        """
        # Patterns to identify mathematical expressions in the context
        math_patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+',  # e.g., "5+3=8"
            r'equals?\s*\d+',
            r'result\s*(?:is|=)\s*\d+',
            r'sum\s*(?:is|=)\s*\d+',
            r'total\s*(?:is|=)\s*\d+'
        ]
        
        context_lower = context.lower()
        for pattern in math_patterns:
            matches = re.findall(pattern, context_lower)
            if matches:
                return f"{matches[0]}"
        
        return "No mathematical calculations found in the provided source material."

    def _prepare_context(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """
        Prepares a combined context string from relevant chunks for the LLM.
        Limits the context length for free models.
        """
        context_parts = []
        total_length = 0
        max_context_length = 1000  # Limit context size for free models

        for chunk, score in relevant_chunks:
            chunk_text = chunk['text']
            chunk_length = len(chunk_text.split())

            if total_length + chunk_length > max_context_length:
                break

            context_parts.append(chunk_text)
            total_length += chunk_length

        return " ".join(context_parts)

    def _generate_fallback_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict = None) -> str:
        """
        Generates a rule-based fallback answer if the LLM confidence is low or an error occurs.
        Attempts to extract key information based on common query types.
        """
        if not relevant_chunks:
            return "I couldn't find relevant information in the provided policy documents to answer this question."

        query_lower = query.lower()
        best_chunk = relevant_chunks[0][0]['text']

        # Rule-based extraction for common insurance queries
        if any(word in query_lower for word in ['grace period', 'payment', 'premium']):
            time_matches = re.findall(r'\b(?:\d+\s*(?:days?|months?|years?))\b', best_chunk, re.IGNORECASE)
            if time_matches:
                return f"According to the policy, the grace period is {time_matches[0]}. {best_chunk[:100]}..."

        if any(word in query_lower for word in ['waiting period', 'wait']):
            time_matches = re.findall(r'\b(?:\d+\s*(?:days?|months?|years?))\b', best_chunk, re.IGNORECASE)
            if time_matches:
                return f"The waiting period is {time_matches[0]}. {best_chunk[:100]}..."

        if any(word in query_lower for word in ['cover', 'coverage', 'include']):
            if any(word in best_chunk.lower() for word in ['yes', 'covered', 'include']):
                return f"Yes, this is covered. {best_chunk[:150]}..."
            elif any(word in best_chunk.lower() for word in ['no', 'not covered', 'exclude']):
                return f"No, this is not covered. {best_chunk[:150]}..."

        # Default response using the best matching chunk
        return f"Based on the policy information: {best_chunk[:200]}..."

    def _post_process_answer(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """
        Post-processes the generated answer by cleaning it and ensuring proper formatting.
        """
        answer = answer.strip()

        if not answer:
            return self._generate_fallback_answer("", relevant_chunks)

        # Clean escape characters and normalize whitespace
        answer = clean_escape_characters(answer)
        answer = re.sub(r'\s+', ' ', answer)

        # Ensure answer ends with a period
        if not answer.endswith('.'):
            answer += '.'

        return answer

    def extract_reasoning(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        """
        Extracts reasoning and confidence metrics for the generated answer.
        Provides a fixed confidence score for free models.
        """
        reasoning = {
            'confidence': 0.8,  # Fixed confidence for open-source models
            'source_chunks': [chunk['text'][:100] + "..." for chunk, _ in relevant_chunks[:3]],
            'reasoning': f"Answer derived from {len(relevant_chunks)} relevant policy sections using open-source AI models"
        }
        return reasoning