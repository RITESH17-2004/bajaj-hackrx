import re
from typing import List, Dict, Tuple

class SimpleAnswerEngine:
    """
    A simple, rule-based answer engine that works without any paid APIs.
    Designed to work with any type of document.
    """
    
    def __init__(self):
        self.general_patterns = {
            'time_period': {
                'keywords': ['period', 'time', 'duration', 'deadline', 'due', 'expires'],
                'patterns': [
                    r'period of (\d+\s*(?:days?|months?|years?))',
                    r'(\d+\s*(?:days?|months?|years?))\s+(?:period|time|duration)',
                    r'within (\d+\s*(?:days?|months?|years?))',
                    r'after (\d+\s*(?:days?|months?|years?))'
                ]
            },
            'eligibility': {
                'keywords': ['eligible', 'qualify', 'requirement', 'criteria', 'condition'],
                'patterns': [
                    r'eligible.*(?:if|when|after)',
                    r'qualify.*(?:if|when|after)',
                    r'requirement.*(?:is|includes)',
                    r'must.*(?:be|have|meet)'
                ]
            },
            'inclusion': {
                'keywords': ['include', 'cover', 'contain', 'comprise', 'feature'],
                'patterns': [
                    r'(?:includes?|covers?|contains?)',
                    r'(?:is\s+)?(?:not\s+)?(?:included|covered|contained)',
                    r'(?:features?|comprises?)'
                ]
            },
            'amounts': {
                'keywords': ['amount', 'cost', 'price', 'fee', 'charge', 'rate'],
                'patterns': [
                    r'(?:\$|Rs\.?|₹)\s*\d+(?:,\d{3})*(?:\.\d{2})?',
                    r'\d+(?:\.\d+)?%',
                    r'rate.*(?:is|of).*\d+',
                    r'(?:costs?|fees?|charges?).*\d+'
                ]
            }
        }
    
    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        if not relevant_chunks:
            return "No relevant information found in the provided documents to answer this question."
        
        query_lower = query.lower()
        best_answer = None
        
        # Try to match specific question patterns
        for category, config in self.general_patterns.items():
            if any(keyword in query_lower for keyword in config['keywords']):
                answer = self._extract_specific_answer(query, relevant_chunks, category, config)
                if answer:
                    best_answer = answer
                    break
        
        # If no specific pattern matched, use general approach
        if not best_answer:
            best_answer = self._generate_general_answer(query, relevant_chunks)
        
        return best_answer
    
    def _extract_specific_answer(self, query: str, chunks: List[Tuple[Dict, float]], category: str, config: Dict) -> str:
        query_lower = query.lower()
        
        for chunk, score in chunks:
            text = chunk['text']
            text_lower = text.lower()
            
            # Look for specific patterns
            for pattern in config['patterns']:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    if category == 'time_period':
                        return f"The time period is {matches[0]}. {text[:150]}..."
                    
                    elif category == 'eligibility':
                        return f"The requirements are: {text[:200]}..."
                    
                    elif category == 'amounts':
                        return f"The amount is {matches[0]}. {text[:150]}..."
            
            # For inclusion questions, look for yes/no indicators
            if category == 'inclusion':
                if any(word in text_lower for word in ['yes', 'included', 'covered', 'contains']):
                    return f"Yes, this is included. {text[:200]}..."
                elif any(word in text_lower for word in ['no', 'not included', 'excluded', 'except']):
                    return f"No, this is not included. {text[:200]}..."
        
        return None
    
    def _generate_general_answer(self, query: str, chunks: List[Tuple[Dict, float]]) -> str:
        if not chunks:
            return "No relevant information found in the provided documents."
        
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
            amounts = re.findall(r'(?:\$|Rs\.?|₹)?\s*\d+(?:,\d{3})*(?:\.\d{2})?|(?:\d+(?:\.\d+)?%)', best_chunk)
            if amounts:
                return f"The amount is {amounts[0]}. {best_chunk[:150]}..."
        
        elif any(word in query_lower for word in ['when', 'time', 'period', 'duration']):
            # Look for time periods
            time_periods = re.findall(r'\b\d+\s*(?:days?|months?|years?)\b', best_chunk, re.IGNORECASE)
            if time_periods:
                return f"The time period is {time_periods[0]}. {best_chunk[:150]}..."
        
        # Default: return the most relevant content with more detail
        return f"{best_chunk[:300]}..."
    
    def extract_reasoning(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        return {
            'confidence': 0.75,  # Fixed confidence for rule-based system
            'source_chunks': [chunk['text'][:100] + "..." for chunk, _ in relevant_chunks[:3]],
            'reasoning': "Answer generated using rule-based pattern matching and keyword extraction from document content"
        }