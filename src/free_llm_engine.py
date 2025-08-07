from transformers import pipeline
import torch
from typing import List, Dict, Tuple
import re
import asyncio
import functools
import logging
from src.text_util import clean_escape_characters

class FreeLLMEngine:
    def __init__(self, device: str = "cpu"):
        logging.info("Loading free LLM model (this may take a few minutes first time)...")

        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad",
            device=0 if device == "cuda" else -1  # 0 for GPU, -1 for CPU
        )

        self.text_generator = pipeline(
            "text-generation",
            model="gpt2",
            max_length=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256,
            device=0 if device == "cuda" else -1
        )

        logging.info("Free LLM models loaded successfully!")

    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(self._generate_answer_sync, query, relevant_chunks, query_intent)
        )

    def _generate_answer_sync(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        if not relevant_chunks:
            return "I couldn't find relevant information in the provided policy documents to answer this question."

        # Combine top relevant chunks as context
        context = self._prepare_context(relevant_chunks[:3])
        
        # Check for mathematical content - handle specially
        content_type = query_intent.get('content_type', 'unknown')
        if content_type == 'mathematical':
            return self._handle_mathematical_content(query, context, relevant_chunks)

        try:
            # Use the question-answering pipeline
            result = self.qa_pipeline(
                question=query,
                context=context,
                max_answer_len=100
            )

            answer = result['answer']
            confidence = result['score']

            # If confidence is too low, try text generation approach
            if confidence < 0.3:
                answer = self._generate_fallback_answer(query, relevant_chunks, query_intent)

            return self._post_process_answer(answer, relevant_chunks)

        except Exception as e:
            logging.error(f"Error with free LLM: {e}")
            return self._generate_fallback_answer(query, relevant_chunks, query_intent)
    
    def _handle_mathematical_content(self, query: str, context: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """Handle mathematical questions by reporting exact content from source"""
        import re
        
        # Look for mathematical expressions in the context
        math_patterns = [
            r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+',  # "5+3=8"
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
        
        # If no mathematical expressions found
        return "No mathematical calculations found in the provided source material."

    def _prepare_context(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        context_parts = []
        total_length = 0
        max_context_length = 1000  # Limit for free models

        for chunk, score in relevant_chunks:
            chunk_text = chunk['text']
            chunk_length = len(chunk_text.split())

            if total_length + chunk_length > max_context_length:
                break

            context_parts.append(chunk_text)
            total_length += chunk_length

        return " ".join(context_parts)

    def _generate_fallback_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict = None) -> str:
        if not relevant_chunks:
            return "I couldn't find relevant information in the provided policy documents to answer this question."

        # Use rule-based approach for common insurance queries
        query_lower = query.lower()
        best_chunk = relevant_chunks[0][0]['text']

        # Extract key information based on query type
        if any(word in query_lower for word in ['grace period', 'payment', 'premium']):
            # Look for time periods in the text
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

        # Default response with best matching chunk
        return f"Based on the policy information: {best_chunk[:200]}..."

    def _post_process_answer(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        answer = answer.strip()

        if not answer:
            return self._generate_fallback_answer("", relevant_chunks)

        # Clean escape characters
        answer = clean_escape_characters(answer)

        # Clean up the answer
        answer = re.sub(r'\s+', ' ', answer)

        if not answer.endswith('.'):
            answer += '.'

        return answer

    def extract_reasoning(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        reasoning = {
            'confidence': 0.8,  # Fixed confidence for free model
            'source_chunks': [chunk['text'][:100] + "..." for chunk, _ in relevant_chunks[:3]],
            'reasoning': f"Answer derived from {len(relevant_chunks)} relevant policy sections using free AI models"
        }
        return reasoning
