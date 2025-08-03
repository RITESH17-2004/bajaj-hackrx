import os
from typing import List, Dict, Tuple
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Try to import Mistral, but fall back to free models if not available
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

try:
    from .free_llm_engine import FreeLLMEngine
    FREE_LLM_AVAILABLE = True
except ImportError:
    FREE_LLM_AVAILABLE = False

from .simple_answer_engine import SimpleAnswerEngine

class DecisionEngine:
    def __init__(self):
        self.mistral_key = os.getenv('MISTRAL_API_KEY', 'your-mistral-api-key-here')
        self.use_mistral = (MISTRAL_AVAILABLE and 
                           self.mistral_key and 
                           self.mistral_key != 'your-mistral-api-key-here')
        
        if self.use_mistral:
            self.client = Mistral(api_key=self.mistral_key)
            self.max_context_length = 4000
            self.temperature = 0.1
            print("Using Mistral AI for LLM processing")
        else:
            if FREE_LLM_AVAILABLE:
                try:
                    self.free_llm = FreeLLMEngine()
                    print("Using free Hugging Face models for LLM processing")
                except Exception as e:
                    print(f"Failed to load Hugging Face models: {e}")
                    self.simple_engine = SimpleAnswerEngine()
                    print("Using simple rule-based engine for LLM processing")
            else:
                self.simple_engine = SimpleAnswerEngine()
                print("Using simple rule-based engine for LLM processing")
        
        self.max_context_length = 4000
        self.temperature = 0.1
    
    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        if self.use_mistral:
            return await self._generate_mistral_answer(query, relevant_chunks, query_intent)
        elif hasattr(self, 'free_llm'):
            return await self.free_llm.generate_answer(query, relevant_chunks, query_intent)
        elif hasattr(self, 'simple_engine'):
            return await self.simple_engine.generate_answer(query, relevant_chunks, query_intent)
        else:
            return self._generate_fallback_answer(query, relevant_chunks)
    
    async def _generate_mistral_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        context = self._prepare_context(relevant_chunks)
        prompt = self._build_prompt(query, context, query_intent)
        
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.complete(
                model="mistral-medium",
                messages=messages,
                temperature=self.temperature,
                max_tokens=100
            )
            
            answer = response.choices[0].message.content.strip()
            return self._post_process_answer(answer, relevant_chunks)
            
        except Exception as e:
            print(f"Mistral API error: {e}")
            return self._generate_fallback_answer(query, relevant_chunks)
    
    def _get_system_prompt(self) -> str:
        return """You are an expert insurance policy analyst. Provide concise, professional answers with essential details only.

STRICT RULES:
1. MAXIMUM 50 words per answer - be direct and focused
2. Include key numbers and timeframes naturally (avoid over-formatting)
3. Include only the most important conditions - not every detail
4. Use clean, professional language without extra formatting
5. Structure: [Direct answer] + [key condition if essential]
6. For yes/no questions: Answer directly with main requirement only
7. For time periods: State timeframe with minimal context
8. For definitions: Give core definition with key specifications only
9. NO brackets, bullet points, or special formatting
10. Base answers strictly on document context but keep focused

Example: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."""
    
    def _build_prompt(self, query: str, context: str, query_intent: Dict) -> str:
        intent_guidance = ""
        response_template = ""
        
        if query_intent['type'] == 'coverage':
            intent_guidance = "State clearly what is covered or not covered with key conditions only."
            response_template = "Answer directly: Yes/No + what's covered + main eligibility requirement."
        elif query_intent['type'] == 'timing':
            intent_guidance = "Provide exact timeframe with essential context only."
            response_template = "State the timeframe clearly with minimal necessary context."
        elif query_intent['type'] == 'information':
            intent_guidance = "Provide essential details and key requirements only."
            response_template = "Give the main definition or details with key specifications."
        else:
            intent_guidance = "Provide focused answer with essential details only."
            response_template = "Answer directly with key information."
        
        prompt = f"""Answer this question using the policy context. Provide a professional, complete response with specific details.

{intent_guidance}

{response_template}

Policy Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _prepare_context(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        context_parts = []
        total_length = 0
        
        # Increase context limit for detailed answers
        max_context_for_detailed_answers = min(self.max_context_length + 1000, 5000)
        
        for chunk, score in relevant_chunks:
            chunk_text = chunk['text']
            chunk_length = len(chunk_text.split())
            
            if total_length + chunk_length > max_context_for_detailed_answers:
                break
            
            # Clean and enhance chunk text for better LLM processing
            enhanced_chunk = self._enhance_chunk_for_context(chunk_text)
            context_parts.append(f"Policy Section: {enhanced_chunk}")
            total_length += chunk_length
        
        return "\n\n".join(context_parts)
    
    def _enhance_chunk_for_context(self, chunk_text: str) -> str:
        """Clean chunk text for better LLM processing without special formatting"""
        import re
        
        # Just clean the text without adding highlighting that might confuse the LLM
        # Remove extra whitespace and clean formatting
        chunk_text = re.sub(r'\s+', ' ', chunk_text)
        chunk_text = chunk_text.strip()
        
        return chunk_text
    
    def _post_process_answer(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        answer = answer.strip()
        
        if not answer or answer.lower().startswith("i don't know") or "not found" in answer.lower():
            return self._generate_fallback_answer("", relevant_chunks)
        
        # Clean up spacing
        answer = re.sub(r'\s+', ' ', answer)
        
        # Validate and enhance answer completeness
        answer = self._validate_and_enhance_answer(answer, relevant_chunks)
        
        if not answer.endswith('.'):
            answer += '.'
        
        return answer
    
    def _validate_and_enhance_answer(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """Validate answer completeness and specificity according to target style"""
        import re
        
        # Check word count (target: 20-50 words)
        word_count = len(answer.split())
        
        # If answer is too long (> 60 words), trim while preserving key information
        if word_count > 60:
            answer = self._trim_long_answer(answer)
        
        # Only enhance if extremely short (< 10 words) and clearly incomplete
        elif word_count < 10 and not any(word in answer.lower() for word in ['yes', 'no', 'days', 'months', 'years', '%']):
            if relevant_chunks:
                answer = self._enhance_short_answer(answer, relevant_chunks)
        
        # Ensure numbers are in proper format (written + numeric)
        answer = self._format_numbers_professionally(answer)
        
        return answer
    
    def _enhance_short_answer(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """Enhance short answers with additional relevant details"""
        if not relevant_chunks:
            return answer
        
        # Extract additional details from the best chunk
        best_chunk = relevant_chunks[0][0]['text']
        
        # Look for conditions, limitations, or additional details
        import re
        conditions = re.findall(r'(provided|subject to|limited to|excluding|including).*?[.;]', best_chunk, re.IGNORECASE)
        
        if conditions and len(conditions[0]) < 100:  # Only add if reasonable length
            answer += f" {conditions[0].strip()}"
        
        return answer
    
    def _trim_long_answer(self, answer: str) -> str:
        """Trim long answers while preserving essential information"""
        sentences = answer.split('. ')
        
        # Keep first sentence (main answer) and most important details
        if len(sentences) > 2:
            # Prioritize sentences with numbers, percentages, or key terms
            key_terms = ['days', 'months', 'years', '%', 'provided', 'covered', 'excluded', 'limited']
            
            main_sentence = sentences[0]
            important_sentences = []
            
            for sentence in sentences[1:]:
                if any(term in sentence.lower() for term in key_terms) and len(important_sentences) < 2:
                    important_sentences.append(sentence)
            
            answer = main_sentence + '. ' + '. '.join(important_sentences)
            if not answer.endswith('.'):
                answer += '.'
        
        return answer
    
    def _format_numbers_professionally(self, answer: str) -> str:
        """Format numbers in professional style naturally"""
        import re
        
        # Only format key time periods and important numbers
        key_numbers = {
            'thirty days': 'thirty days',  # Keep natural
            'thirty-six months': 'thirty-six (36) months',  # Only format complex numbers
            'twenty-four months': 'twenty-four (24) months',
            'two years': 'two (2) years'
        }
        
        for written, formatted in key_numbers.items():
            if written in answer.lower():
                # Only add numeric if it adds value (complex numbers)
                if '(' not in answer and len(written.split()[0]) > 5:  # Only for longer numbers
                    answer = re.sub(re.escape(written), formatted, answer, flags=re.IGNORECASE)
        
        return answer
    
    def _generate_fallback_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        if not relevant_chunks:
            return "I couldn't find relevant information in the provided policy documents to answer this question."
        
        best_chunk = relevant_chunks[0][0]['text']
        
        # Extract specific facts instead of raw text for professional fallback
        import re
        
        # Look for specific numbers, timeframes, or amounts
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\s*(days?|months?|years?|%|\$|Rs\.?)\b', best_chunk, re.IGNORECASE)
        if numbers:
            number, unit = numbers[0]
            return f"According to the policy, the specified {unit.lower()} is {number} {unit.lower()}."
        
        # Look for yes/no coverage statements
        if any(word in best_chunk.lower() for word in ['covered', 'includes', 'benefits']):
            return f"Based on the policy terms, this appears to be covered. {best_chunk[:100]}..."
        elif any(word in best_chunk.lower() for word in ['excluded', 'not covered', 'except']):
            return f"Based on the policy terms, this appears to be excluded. {best_chunk[:100]}..."
        
        # Default professional fallback
        return f"Based on the available policy information: {best_chunk[:120]}..."
    
    def extract_reasoning(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        reasoning = {
            'confidence': self._calculate_confidence(answer, relevant_chunks),
            'source_chunks': [chunk['text'][:100] + "..." for chunk, _ in relevant_chunks[:3]],
            'reasoning': f"Answer derived from {len(relevant_chunks)} relevant policy sections"
        }
        
        return reasoning
    
    def _calculate_confidence(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> float:
        if not relevant_chunks:
            return 0.1
        
        avg_relevance = sum(score for _, score in relevant_chunks) / len(relevant_chunks)
        
        answer_indicators = [
            'specifically states',
            'according to',
            'the policy',
            'section',
            'clause'
        ]
        
        confidence_boost = sum(0.1 for indicator in answer_indicators if indicator in answer.lower())
        
        base_confidence = min(avg_relevance, 0.8)
        final_confidence = min(base_confidence + confidence_boost, 0.95)
        
        return round(final_confidence, 2)