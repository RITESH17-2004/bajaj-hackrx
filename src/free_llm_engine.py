from transformers import pipeline
import torch
from typing import List, Dict, Tuple
import re

class FreeLLMEngine:
    def __init__(self):
        print("Loading free LLM model (this may take a few minutes first time)...")
        
        # Use a smaller, free model that works well for Q&A
        self.qa_pipeline = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad",
            tokenizer="distilbert-base-cased-distilled-squad"
        )
        
        # For text generation, use a smaller GPT-2 model
        self.text_generator = pipeline(
            "text-generation",
            model="gpt2",
            max_length=200,
            do_sample=True,
            temperature=0.7,
            pad_token_id=50256
        )
        
        print("Free LLM models loaded successfully!")
    
    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        if not relevant_chunks:
            return "I couldn't find relevant information in the provided policy documents to answer this question."
        
        # Combine top relevant chunks as context
        context = self._prepare_context(relevant_chunks[:3])
        
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
                answer = self._generate_fallback_answer(query, context)
            
            return self._post_process_answer(answer, relevant_chunks)
            
        except Exception as e:
            print(f"Error with free LLM: {e}")
            return self._generate_fallback_answer(query, relevant_chunks)
    
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
    
    def _generate_fallback_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
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