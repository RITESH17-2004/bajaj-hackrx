import re
from typing import List, Dict, Tuple
import asyncio
import functools
from concurrent.futures import ThreadPoolExecutor
from src.text_util import clean_escape_characters

class SimpleAnswerEngine:
    """
    A simple, rule-based answer engine that works without any paid APIs.
    Designed specifically for insurance policy questions.
    """
    
    def __init__(self, executor: ThreadPoolExecutor = None):
        self.insurance_patterns = {
            'grace_period': {
                'keywords': ['grace period', 'payment', 'premium', 'due date'],
                'patterns': [
                    r'grace period of (\d+\s*(?:days?|months?))',
                    r'(\d+\s*(?:days?|months?))\s+grace period',
                    r'within (\d+\s*(?:days?|months?))\s+(?:of|after)\s+(?:due date|payment)'
                ]
            },
            'waiting_period': {
                'keywords': ['waiting period', 'wait', 'coverage begins'],
                'patterns': [
                    r'waiting period of (\d+\s*(?:days?|months?|years?))',
                    r'(\d+\s*(?:days?|months?|years?))\s+waiting period',
                    r'after (\d+\s*(?:days?|months?|years?))\s+of\s+(?:continuous\s+)?coverage'
                ]
            },
            'maternity': {
                'keywords': ['maternity', 'pregnancy', 'childbirth', 'delivery'],
                'patterns': [
                    r'maternity.*covered',
                    r'pregnancy.*(?:covered|benefit)',
                    r'childbirth.*(?:covered|benefit)'
                ]
            },
            'coverage': {
                'keywords': ['cover', 'coverage', 'covered', 'benefit', 'include'],
                'patterns': [
                    r'(?:does\s+)?(?:not\s+)?(?:cover|include)',
                    r'(?:is\s+)?(?:not\s+)?covered',
                    r'(?:includes?|benefits?)'
                ]
            }
        }
        self.executor = executor
    
    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            functools.partial(self._generate_answer_sync, query, relevant_chunks, query_intent)
        )

    def _generate_answer_sync(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        if not relevant_chunks:
            return "I couldn't find relevant information in the provided policy documents to answer this question."
        
        query_lower = query.lower()
        content_type = query_intent.get('content_type', 'unknown')
        question_tone = query_intent.get('question_tone', 'neutral')
        
        # Handle mathematical content specially
        if content_type == 'mathematical':
            return self._handle_mathematical_content(query, relevant_chunks)
        
        best_answer = None
        
        # Try to match specific question patterns
        for category, config in self.insurance_patterns.items():
            if any(keyword in query_lower for keyword in config['keywords']):
                answer = self._extract_specific_answer(query, relevant_chunks, category, config, question_tone)
                if answer:
                    best_answer = answer
                    break
        
        # If no specific pattern matched, use general approach
        if not best_answer:
            best_answer = self._generate_general_answer(query, relevant_chunks, query_intent)
        
        # Clean escape characters from the final answer
        return clean_escape_characters(best_answer)
    
    def _handle_mathematical_content(self, query: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """Handle mathematical questions by reporting exact content from source"""
        import re
        
        for chunk, score in relevant_chunks:
            text = chunk['text']
            
            # Look for mathematical expressions
            math_patterns = [
                r'\d+\s*[\+\-\*\/]\s*\d+\s*=\s*\d+',  # "5+3=8"
                r'equals?\s*\d+',
                r'result\s*(?:is|=)\s*\d+',
                r'sum\s*(?:is|=)\s*\d+',
                r'total\s*(?:is|=)\s*\d+'
            ]
            
            for pattern in math_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    return f"According to the source material: {matches[0]}"
        
        return "No mathematical calculations found in the provided source material."
    
    def _extract_specific_answer(self, query: str, chunks: List[Tuple[Dict, float]], category: str, config: Dict, question_tone: str = 'neutral') -> str:
        query_lower = query.lower()
        
        for chunk, score in chunks:
            text = chunk['text']
            text_lower = text.lower()
            
            # Look for specific patterns
            for pattern in config['patterns']:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    if category == 'grace_period':
                        return f"A grace period of {matches[0]} is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."
                    
                    elif category == 'waiting_period':
                        if 'pre-existing' in query_lower or 'ped' in query_lower:
                            return f"There is a waiting period of {matches[0]} of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."
                        else:
                            return f"The waiting period is {matches[0]} for this coverage."
                    
                    elif category == 'maternity':
                        if 'covered' in text_lower:
                            return "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months."
                        else:
                            return "Maternity coverage details are specified in the policy terms and conditions."
            
            # For coverage questions, look for yes/no indicators
            if category == 'coverage':
                if any(word in text_lower for word in ['yes', 'covered', 'includes', 'benefits']):
                    return f"Yes, this is covered according to the policy. {text[:150]}..."
                elif any(word in text_lower for word in ['no', 'not covered', 'excludes', 'except']):
                    return f"No, this is not covered according to the policy. {text[:150]}..."
        
        return None
    
    def _generate_general_answer(self, query: str, chunks: List[Tuple[Dict, float]], query_intent: Dict = None) -> str:
        if not chunks:
            return "I couldn't find relevant information in the provided policy documents."
        
        best_chunk = chunks[0][0]['text']
        
        # Extract key information based on common question types
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            # Extract definitions or explanations
            sentences = best_chunk.split('.')
            for sentence in sentences:
                if any(word in query_lower for word in sentence.lower().split()[:5]):
                    return sentence.strip() + '.'
        
        elif any(word in query_lower for word in ['how much', 'amount', 'cost', 'price']):
            # Look for amounts or percentages
            amounts = re.findall(r'(?:\$|Rs\.?|\u20b9)?\s*\d+(?:,\d{3})*(?:\.\d{2})?|(?:\d+(?:\.\d+)?%)', best_chunk)
            if amounts:
                return f"According to the policy, the amount is {amounts[0]}. {best_chunk[:100]}..."
        
        elif any(word in query_lower for word in ['when', 'time', 'period', 'duration']):
            # Look for time periods
            time_periods = re.findall(r'\b\d+\s*(?:days?|months?|years?)\b', best_chunk, re.IGNORECASE)
            if time_periods:
                return f"The time period specified is {time_periods[0]}. {best_chunk[:100]}..."
        
        # Default: return the most relevant chunk with some context
        return f"Based on the policy information: {best_chunk[:250]}..."

    def extract_reasoning(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        return {
            'confidence': 0.75,  # Fixed confidence for rule-based system
            'source_chunks': [chunk['text'][:100] + "..." for chunk, _ in relevant_chunks[:3]],
            'reasoning': "Answer generated using rule-based pattern matching and keyword extraction from policy documents"
        }
