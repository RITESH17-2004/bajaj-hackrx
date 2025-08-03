import os
import asyncio
from typing import List, Dict, Tuple
import json
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Mistral commented out - using Gemini Flash only
# try:
#     from mistralai import Mistral
#     MISTRAL_AVAILABLE = True
# except ImportError:
#     MISTRAL_AVAILABLE = False
MISTRAL_AVAILABLE = False

# Try to import Google Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from .free_llm_engine import FreeLLMEngine
    FREE_LLM_AVAILABLE = True
except ImportError:
    FREE_LLM_AVAILABLE = False

from .simple_answer_engine import SimpleAnswerEngine

class DecisionEngine:
    def __init__(self):
        # Commented out Mistral setup
        # self.mistral_key = os.getenv('MISTRAL_API_KEY', 'your-mistral-api-key-here')
        # self.use_mistral = (MISTRAL_AVAILABLE and 
        #                    self.mistral_key and 
        #                    self.mistral_key != 'your-mistral-api-key-here')
        
        # Setup Gemini Flash with optimizations
        self.gemini_key = os.getenv('GEMINI_API_KEY', 'your-gemini-api-key-here')
        self.use_gemini = (GEMINI_AVAILABLE and 
                          self.gemini_key and 
                          self.gemini_key != 'your-gemini-api-key-here')
        
        if self.use_gemini:
            genai.configure(api_key=self.gemini_key)
            
            # Use fastest Gemini model from env or default
            model_name = os.getenv('GEMINI_MODEL', 'gemini-2.0-flash-exp')
            self.gemini_model = genai.GenerativeModel(model_name)
            
            # Performance optimizations
            self.max_context_length = 1000000  # Gemini Flash 1M context window
            self.temperature = float(os.getenv('GEMINI_TEMPERATURE', '0.1'))
            self.max_tokens = int(os.getenv('GEMINI_MAX_TOKENS', '200'))  # Increased for detailed answers
            self.timeout = int(os.getenv('GEMINI_TIMEOUT', '30'))
            
            print(f"Using {model_name} for LLM processing with optimizations")
            print(f"Max tokens: {self.max_tokens}, Temperature: {self.temperature}, Timeout: {self.timeout}s")
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
        
        self.max_context_length = self.max_context_length if hasattr(self, 'max_context_length') else 4000
        self.temperature = 0.1
    
    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        if self.use_gemini:
            return await self._generate_gemini_answer(query, relevant_chunks, query_intent)
        # Commented out Mistral
        # elif self.use_mistral:
        #     return await self._generate_mistral_answer(query, relevant_chunks, query_intent)
        elif hasattr(self, 'free_llm'):
            return await self.free_llm.generate_answer(query, relevant_chunks, query_intent)
        elif hasattr(self, 'simple_engine'):
            return await self.simple_engine.generate_answer(query, relevant_chunks, query_intent)
        else:
            return self._generate_fallback_answer(query, relevant_chunks)
    
    async def _generate_gemini_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        context = self._prepare_context(relevant_chunks)
        prompt = self._build_prompt(query, context, query_intent)
        
        try:
            # Combine system prompt and user prompt for Gemini
            full_prompt = f"{self._get_system_prompt()}\n\n{prompt}"
            
            response = await asyncio.create_task(
                self._call_gemini_async(full_prompt)
            )
            
            answer = response.strip()
            return self._post_process_answer(answer, relevant_chunks)
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return self._generate_fallback_answer(query, relevant_chunks)
    
    async def _call_gemini_async(self, prompt: str) -> str:
        """Optimized async wrapper for Gemini API call"""
        loop = asyncio.get_event_loop()
        
        def make_request():
            return self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    candidate_count=1,  # Single response for speed
                    stop_sequences=None,  # No stop sequences for efficiency
                )
            ).text
        
        try:
            # Use timeout for faster failure handling
            response = await asyncio.wait_for(
                loop.run_in_executor(None, make_request),
                timeout=self.timeout
            )
            return response
        except asyncio.TimeoutError:
            print(f"Gemini API timeout after {self.timeout}s")
            raise Exception("API timeout - try reducing document size or context")

    # Commented out Mistral method
    # async def _generate_mistral_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
    #     context = self._prepare_context(relevant_chunks)
    #     prompt = self._build_prompt(query, context, query_intent)
    #     
    #     try:
    #         messages = [
    #             {"role": "system", "content": self._get_system_prompt()},
    #             {"role": "user", "content": prompt}
    #         ]
    #         
    #         response = self.client.chat.complete(
    #             model="mistral-medium",
    #             messages=messages,
    #             temperature=self.temperature,
    #             max_tokens=100
    #         )
    #         
    #         answer = response.choices[0].message.content.strip()
    #         return self._post_process_answer(answer, relevant_chunks)
    #         
    #     except Exception as e:
    #         print(f"Mistral API error: {e}")
    #         return self._generate_fallback_answer(query, relevant_chunks)
    
    def _get_system_prompt(self) -> str:
        return """You are an expert policy analyst. Provide detailed, accurate answers with specific information.

RESPONSE GUIDELINES:
1. Give comprehensive answers with all relevant details (50-80 words)
2. Include specific numbers, timeframes, percentages, and conditions
3. For yes/no questions: Start with yes/no, then provide full explanation with conditions
4. Include eligibility criteria, limits, and important conditions
5. Use precise language and exact terminology from the content
6. Provide complete information in a single response
7. Structure: Direct answer → Specific details → Key conditions/limitations
8. Be thorough but clear and well-organized

Example: "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period."""
    
    def _build_prompt(self, query: str, context: str, query_intent: Dict) -> str:
        intent_guidance = ""
        response_template = ""
        
        if query_intent['type'] == 'coverage':
            intent_guidance = "Provide comprehensive coverage details including eligibility, conditions, and limitations."
            response_template = "Start with yes/no, then explain what's covered, eligibility criteria, and any limits or conditions."
        elif query_intent['type'] == 'timing':
            intent_guidance = "Give specific timeframes with context and conditions."
            response_template = "State the exact time period and explain what it applies to, including any eligibility requirements."
        elif query_intent['type'] == 'information':
            intent_guidance = "Provide detailed explanation with specific criteria and conditions."
            response_template = "Give comprehensive information including definitions, requirements, and applicable conditions."
        else:
            intent_guidance = "Provide thorough answer with all relevant details."
            response_template = "Include specific numbers, conditions, eligibility criteria, and limitations where applicable."
        
        prompt = f"""Answer this question thoroughly with all relevant details.

{intent_guidance}

{response_template}

Context:
{context}

Question: {query}

Provide a comprehensive answer:"""
        
        return prompt
    
    def _prepare_context(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        context_parts = []
        total_length = 0
        
        # Increase context limit for detailed answers
        max_context_for_detailed_answers = min(self.max_context_length + 2000, 8000)
        
        for chunk, score in relevant_chunks:
            chunk_text = chunk['text']
            chunk_length = len(chunk_text.split())
            
            if total_length + chunk_length > max_context_for_detailed_answers:
                break
            
            # Clean and enhance chunk text for better LLM processing
            enhanced_chunk = self._enhance_chunk_for_context(chunk_text)
            context_parts.append(enhanced_chunk)  # Remove "Document Section:" prefix
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
            return f"The specified {unit.lower()} is {number} {unit.lower()}."
        
        # Look for yes/no coverage statements
        if any(word in best_chunk.lower() for word in ['covered', 'includes', 'benefits']):
            return f"Yes, this is included."
        elif any(word in best_chunk.lower() for word in ['excluded', 'not covered', 'except']):
            return f"No, this is not covered."
        
        # Default clean fallback
        return f"{best_chunk[:100]}..."
    
    def extract_reasoning(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        reasoning = {
            'confidence': self._calculate_confidence(answer, relevant_chunks),
            'source_chunks': [chunk['text'][:100] + "..." for chunk, _ in relevant_chunks[:3]],
            'reasoning': f"Answer derived from {len(relevant_chunks)} relevant document sections"
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