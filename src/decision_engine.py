import os
from typing import List, Dict, Tuple
import json
import re
from dotenv import load_dotenv
import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import logging

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
    def __init__(self, device: str = "cpu", executor: ThreadPoolExecutor = None):
        self.mistral_key = os.getenv('MISTRAL_API_KEY', 'your-mistral-api-key-here')
        self.use_mistral = (MISTRAL_AVAILABLE and 
                           self.mistral_key and 
                           self.mistral_key != 'your-mistral-api-key-here')
        
        if self.use_mistral:
            self.client = Mistral(api_key=self.mistral_key)
            self.max_context_length = 4000
            self.temperature = 0.1
            logging.info("Using Mistral AI for LLM processing")
        else:
            if FREE_LLM_AVAILABLE:
                try:
                    self.free_llm = FreeLLMEngine(device=device)
                    logging.info("Using free Hugging Face models for LLM processing")
                except Exception as e:
                    logging.error(f"Failed to load Hugging Face models: {e}")
                    self.simple_engine = SimpleAnswerEngine()
                    logging.info("Using simple rule-based engine for LLM processing")
            else:
                self.simple_engine = SimpleAnswerEngine()
                logging.info("Using simple rule-based engine for LLM processing")
        
        self.max_context_length = 4000
        self.temperature = 0.1
        self.executor = executor
    
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
        
        loop = asyncio.get_event_loop()
        
        try:
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
            
            response = await loop.run_in_executor(
                self.executor,
                functools.partial(
                    self.client.chat.complete,
                    model="mistral-small-latest",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1500  # Increase max tokens for multiple answers
                )
            )
            
            answer = response.choices[0].message.content.strip()
            return self._post_process_answer(answer, relevant_chunks)
            
        except Exception as e:
            logging.error(f"Mistral API error: {e}")
            # Implement exponential backoff for rate limiting
            if "429" in str(e) or "Service tier capacity exceeded" in str(e):
                for i in range(3):  # Max 3 retries
                    delay = 2 ** i  # Exponential backoff: 1, 2, 4 seconds
                    logging.warning(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay) # Use asyncio.sleep
                    try:
                        response = await loop.run_in_executor(
                            self.executor,
                            functools.partial(
                                self.client.chat.complete,
                                model="mistral-small-latest",
                                messages=messages,
                                temperature=self.temperature,
                                max_tokens=100
                            )
                        )
                        answer = response.choices[0].message.content.strip()
                        return self._post_process_answer(answer, relevant_chunks)
                    except Exception as retry_e:
                        logging.error(f"Retry failed: {retry_e}")
            return self._generate_fallback_answer(query, relevant_chunks)

    async def generate_answers_in_batch(self, queries: List[str], relevant_chunks_map: Dict[str, List[Tuple[Dict, float]]]) -> List[str]:
        if not self.use_mistral:
            # Fallback to sequential processing if Mistral is not available
            answers = []
            for query in queries:
                relevant_chunks = relevant_chunks_map.get(query, [])
                answer = await self.generate_answer(query, relevant_chunks, {})
                answers.append(answer)
            return answers

        # Combine all questions and contexts into a single prompt
        prompt = self._build_batch_prompt(queries, relevant_chunks_map)

        try:
            messages = [
                {"role": "system", "content": self._get_batch_system_prompt()},
                {"role": "user", "content": prompt}
            ]

            response = await loop.run_in_executor(
                self.executor,
                functools.partial(
                    self.client.chat.complete,
                    model="mistral-small-latest",
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=1500  # Increase max tokens for multiple answers
                )
            )

            response_text = response.choices[0].message.content.strip()
            return self._parse_batch_response(response_text, queries)

        except Exception as e:
            logging.error(f"Mistral API error in batch processing: {e}")
            # Implement exponential backoff for rate limiting
            if "429" in str(e) or "Service tier capacity exceeded" in str(e):
                for i in range(3):  # Max 3 retries
                    delay = 2 ** i  # Exponential backoff: 1, 2, 4 seconds
                    logging.warning(f"Retrying batch in {delay} seconds...")
                    time.sleep(delay)
                    try:
                        response = self.client.chat.complete(
                            model="mistral-small-latest",
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=1500  # Increase max tokens for multiple answers
                        )
                        response_text = response.choices[0].message.content.strip()
                        return self._parse_batch_response(response_text, queries)
                    except Exception as retry_e:
                        logging.error(f"Batch retry failed: {retry_e}")
            # Fallback to sequential processing on error
            answers = []
            for query in queries:
                relevant_chunks = relevant_chunks_map.get(query, [])
                answers.append(self._generate_fallback_answer(query, relevant_chunks))
            return answers

    def _build_batch_prompt(self, queries: List[str], relevant_chunks_map: Dict[str, List[Tuple[Dict, float]]]) -> str:
        prompt_parts = []
        for i, query in enumerate(queries):
            context = self._prepare_context(relevant_chunks_map.get(query, []))
            prompt_parts.append(f"Question {i+1}: {query}\nContext: {context}\n")
        
        return "Answer the following questions based on the provided context for each. Provide a concise and direct answer for each question.\n\n" + "\n".join(prompt_parts)

    def _get_batch_system_prompt(self) -> str:
        return """You are a subject matter expert providing precise answers to a series of questions based strictly on the provided document context for each. For each question, provide a direct and factual answer. If the information is not available in the context, state 'Not specified in document'.

        Respond in the format:
        Answer 1: [Your answer to question 1]
        Answer 2: [Your answer to question 2]
        """

    def _parse_batch_response(self, response_text: str, queries: List[str]) -> List[str]:
        answers = []
        for i, query in enumerate(queries):
            match = re.search(f"Answer {i+1}:(.*?)(?=Answer {i+2}:|$)", response_text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                answers.append(answer)
            else:
                answers.append("Could not parse answer from the model's response.")
        return answers
    
    def _get_system_prompt(self) -> str:
        return """You are a subject matter expert providing precise answers based strictly on the provided document.

RESPONSE RULES:
1. MAXIMUM 50 words - be direct and factual
2. Answer only what the document explicitly states
3. If information is missing, state "Not specified in document"
4. For conflicting information, note "Document contains conflicting information"

ANSWER FORMATS:
- Yes/No questions: Direct answer + primary requirement only
- Timeframes: State period with minimal essential context
- Definitions: Core meaning + critical specifications only
- Numbers/amounts: Include relevant figures and conditions naturally

LANGUAGE REQUIREMENTS:
- Professional, clear prose - no bullets, brackets, or special formatting
- Include only the most critical conditions, not every detail
- Structure: [Direct answer] + [essential qualifier if needed]
- Avoid assumptions or inferences beyond document content

Base all responses strictly on document evidence while maintaining focus and brevity.


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
        max_context_for_detailed_answers = min(self.max_context_length, 1500)
        
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
            return f"According to the policy, the specified {unit.lower()} is {number} {unit.lower()}"
        
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
