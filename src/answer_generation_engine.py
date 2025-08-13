import os
from typing import List, Dict, Tuple

import re
from dotenv import load_dotenv
import asyncio
import functools
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from src.text_cleaner_utils import clean_escape_characters

# Load environment variables from .env file
load_dotenv()

# Attempt to import Mistral AI library; fall back to free models if not available
try:
    from mistralai import Mistral
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

# Attempt to import FreeLLMEngine for local LLM inference
try:
    from .open_source_llm_engine import FreeLLMEngine
    FREE_LLM_AVAILABLE = True
except ImportError:
    FREE_LLM_AVAILABLE = False

from .rule_based_answer_engine import RuleBasedAnswerEngine

class AnswerGenerationEngine:
    """
    The AnswerGenerationEngine is responsible for generating answers to queries based on
    relevant document chunks and a determined query intent. It can utilize either
    the Mistral AI model, a local FreeLLMEngine, or a simple rule-based fallback.
    """
    def __init__(self, device: str = "cpu", executor: ThreadPoolExecutor = None):
        # Retrieve Mistral API key from environment variables
        self.mistral_key = os.getenv('MISTRAL_API_KEY', 'your-mistral-api-key-here')
        # Determine if Mistral AI should be used based on availability and API key
        self.use_mistral = (MISTRAL_AVAILABLE and
                           self.mistral_key and
                           self.mistral_key != 'your-mistral-api-key-here')

        if self.use_mistral:
            self.client = Mistral(api_key=self.mistral_key)
            self.max_context_length = 4000
            self.temperature = 0.0
            logging.info("AnswerGenerationEngine initialized with Mistral AI.")
        else:
            logging.warning("Mistral API key not found or invalid. Attempting fallback LLM.")
            if FREE_LLM_AVAILABLE:
                try:
                    self.free_llm = FreeLLMEngine(device=device)
                    logging.info("AnswerGenerationEngine initialized with Open Source LLM.")
                except Exception as e:
                    logging.error(f"Failed to load Hugging Face models: {e}. Falling back to RuleBasedAnswerEngine.")
                    self.simple_engine = RuleBasedAnswerEngine()
                    logging.info("AnswerGenerationEngine initialized with Rule-Based Engine (Fallback 1).")
            else:
                self.simple_engine = RuleBasedAnswerEngine()
                logging.info("AnswerGenerationEngine initialized with Rule-Based Engine (Fallback 2).")

        self.max_context_length = 4000  # Maximum token length for context fed to LLM
        self.temperature = 0.0          # LLM temperature for deterministic output
        self.executor = executor        # Thread pool executor for async operations

    async def generate_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        """
        Generates an answer to a single query using the configured LLM or fallback.
        """
        if self.use_mistral:
            return await self._generate_mistral_answer(query, relevant_chunks, query_intent)
        elif hasattr(self, 'free_llm'):
            return await self.free_llm.generate_answer(query, relevant_chunks, query_intent)
        elif hasattr(self, 'simple_engine'):
            return await self.simple_engine.generate_answer(query, relevant_chunks, query_intent)
        else:
            return self._generate_fallback_answer(query, relevant_chunks)

    async def _generate_mistral_answer(self, query: str, relevant_chunks: List[Tuple[Dict, float]], query_intent: Dict) -> str:
        """
        Generates an answer using the Mistral AI model.
        Prepares context and prompt, handles API calls, and post-processes the response.
        Includes retry logic for rate limiting.
        """
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
                    max_tokens=1500  # Max tokens for response generation
                )
            )

            answer = response.choices[0].message.content.strip()
            logging.info(f"Raw answer from Mistral LLM: {answer}")
            logging.info("Raw answer from Mistral LLM received.")
            return self._post_process_answer(answer, relevant_chunks)

        except Exception as e:
            logging.error(f"Mistral API error during single query: {e}")
            # Implement exponential backoff for rate limiting or service capacity issues
            if "429" in str(e) or "Service tier capacity exceeded" in str(e):
                for i in range(3):  # Max 3 retries
                    delay = 2 ** i  # Exponential backoff: 1, 2, 4 seconds
                    logging.warning(f"Retrying Mistral single query in {delay} seconds...")
                    await asyncio.sleep(delay)
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
                        logging.error(f"Mistral single query retry failed: {retry_e}")
            return self._generate_fallback_answer(query, relevant_chunks)

    async def generate_answers_in_batch(self, queries: List[str], relevant_chunks_map: Dict[str, List[Tuple[Dict, float]]]) -> List[str]:
        """
        Generates answers for multiple queries in a single batch request to Mistral AI.
        Falls back to sequential processing if Mistral is not available or batch fails.
        """
        if not self.use_mistral:
            # Fallback to sequential processing if Mistral is not available
            answers = []
            for query in queries:
                relevant_chunks = relevant_chunks_map.get(query, [])
                answer = await self.generate_answer(query, relevant_chunks, {{}})
                answers.append(answer)
            return answers

        # Combine all questions and contexts into a single prompt for batch processing
        prompt = self._build_batch_prompt(queries, relevant_chunks_map)

        loop = asyncio.get_event_loop()

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
                    max_tokens=1500  # Max tokens for multiple answers in batch
                )
            )

            response_text = response.choices[0].message.content.strip()
            logging.info("Raw batch answer from Mistral LLM received.")
            return self._parse_batch_response(response_text, queries)

        except Exception as e:
            logging.error(f"Mistral API error in batch processing: {e}")
            # Implement exponential backoff for rate limiting
            if "429" in str(e) or "Service tier capacity exceeded" in str(e):
                for i in range(3):  # Max 3 retries
                    delay = 2 ** i  # Exponential backoff: 1, 2, 4 seconds
                    logging.warning(f"Retrying Mistral batch in {delay} seconds...")
                    time.sleep(delay)
                    try:
                        response = self.client.chat.complete(
                            model="mistral-small-latest",
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=1500
                        )
                        response_text = response.choices[0].message.content.strip()
                        return self._parse_batch_response(response_text, queries)
                    except Exception as retry_e:
                        logging.error(f"Mistral batch retry failed: {retry_e}")
            # Fallback to sequential processing on persistent error
            answers = []
            for query in queries:
                relevant_chunks = relevant_chunks_map.get(query, [])
                answers.append(self._generate_fallback_answer(query, relevant_chunks))
            return answers

    def _build_batch_prompt(self, queries: List[str], relevant_chunks_map: Dict[str, List[Tuple[Dict, float]]]) -> str:
        """
        Constructs a single prompt for batch processing of multiple questions.
        """
        prompt_parts = []
        for i, query in enumerate(queries):
            context = self._prepare_context(relevant_chunks_map.get(query, []))
            prompt_parts.append(f"Question {i+1}: {query}\nContext: {context}\n")

        return "Answer the following questions based on the provided context for each. Provide a concise and direct answer for each question.\n\n" + "\n".join(prompt_parts)

    def _get_batch_system_prompt(self) -> str:
        """
        Returns the system prompt for batch question answering.
        """
        return """You are a subject matter expert providing precise answers to a series of questions based strictly on the provided document context for each. For each question, provide a direct and factual answer. If the information is not available in the context, state 'Not specified in document'.

        Respond in the format:
        Answer 1: [Your answer to question 1]
        Answer 2: [Your answer to question 2]
        """

    def _parse_batch_response(self, response_text: str, queries: List[str]) -> List[str]:
        """
        Parses the batch response from the LLM into individual answers.
        """
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
        """
        Returns the system prompt for single question answering.
        This prompt guides the LLM on how to generate responses based on provided source material.
        """
        return """You are a context-aware AI assistant providing precise answers based solely on the provided source material (documents, images, Excel sheets, PDFs, etc.).

CORE PRINCIPLES:
• Answers must be derived directly from the provided source content. Rephrase as needed for clarity and conciseness, but do not introduce new information.
• Adapt response style based on content type and question complexity.
• Never add external knowledge, calculations, or assumptions.
• If the question is asked in a particular language, respond in that same language while following all other principles.
• Understand and interpret closely related terms (e.g., "PED" and "pre-existing diseases") as referring to the same concept if the context supports it.

SOURCE CONTENT FIDELITY:
• Mathematical operations: Report results exactly as shown (e.g., if source shows “9+5=22”, answer “22”). Do not perform calculations yourself.
• Text content: Accurately convey information from the source. Minor rephrasing for readability is allowed, but the meaning must remain identical.
• Data values: Use specific amounts, dates, percentages from source.
• Source content takes precedence over general factual accuracy if there's a discrepancy.

GROUNDED ANSWERING:
• Only answer questions with information explicitly written in the provided source. Do not guess, assume, or infer objectives, intentions, or consequences. If the answer is not reply in the context to the question
• Never include objectives, purposes, or outcomes unless they are directly stated in the provided content.
• Always use the same language as the source text for factual extraction.

CONTEXT-AWARE RESPONSE ADAPTATION:
The user prompt will specify the appropriate response approach based on content analysis. Follow the provided instruction and template precisely while maintaining these standards:
• Word target: 35-45 words maximum for natural flow.
• Professional tone: Sound knowledgeable but conversational.
• Complete responses: Always end with proper punctuation.
• Prioritize: Direct answers, timeframes, amounts, key conditions from source.

CONTENT PRIORITIZATION:
• Essential: Direct answer, exact values, primary requirements from source.
• Include if space: Key limitations, specific conditions.
• Eliminate: Secondary details, background context, redundant information.

EXCEL FILE HANDLING:
If the answer is extracted from an Excel sheet, do not mention the cell, row, column, or sheet location of the source.

QUALITY STANDARDS:
• Natural, professional writing – not robotic or choppy.
• Focus on one key point per answer.
• Accuracy to source content over general correctness.
• Adapt response complexity to match question tone and content type.
"""

    def _analyze_source_content_type(self, context: str) -> str:
        """Analyzes the type of content within the source material (e.g., mathematical, data, general)."""
        context_lower = context.lower()
        
        # Indicators for mathematical content
        math_indicators = [
            r'\d+\s*[\+\-\\/]\s*\d+\s*=\s*\d+',  # calculations like "5+3=8"
            r'equals?\s*\d+',
            r'result\s*(?:is|=)\s*\d+',
            r'sum\s*(?:is|=)\s*\d+',
            r'total\s*(?:is|=)\s*\d+'
        ]
        
        for pattern in math_indicators:
            if re.search(pattern, context_lower):
                return 'mathematical'
        
        # Indicators for data/numerical content
        data_indicators = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # currency
            r'\d+(?:\.\d+)?%',                # percentages
            r'amount.*\d+',
            r'price.*\d+',
            r'cost.*\d+',
            r'rate.*\d+'
        ]
        
        if any(re.search(pattern, context_lower) for pattern in data_indicators):
            return 'data'
        
        return 'general'

    def _build_prompt(self, query: str, context: str, query_intent: Dict) -> str:
        """
        Constructs the prompt for the LLM, incorporating query, context, and intent analysis.
        """
        # Analyze both query intent and source content type
        content_type = self._analyze_source_content_type(context)
        query_content_type = query_intent.get('content_type', 'unknown')
        question_tone = query_intent.get('question_tone', 'neutral')
        intent_type = query_intent.get('type', 'general')
        
        # Select response approach based on content and question analysis
        response_guidance = self._select_response_approach(
            content_type, query_content_type, question_tone, intent_type
        )
        
        prompt = f"""{response_guidance['instruction']}

Source Material:
{context}

Question: {query}

{response_guidance['template']}"""

        return prompt

    def _select_response_approach(self, content_type: str, query_content_type: str, 
                                question_tone: str, intent_type: str) -> Dict[str, str]:
        """
        Selects the appropriate response approach (instruction and template) for the LLM
        based on the analyzed content type, query content type, question tone, and intent.
        """
        
        # Handle mathematical content
        if content_type == 'mathematical' or query_content_type == 'mathematical':
            return {
                'instruction': 'Report mathematical calculations and results exactly as shown in the source material. Never perform calculations yourself.',
                'template': 'Answer: Report the exact calculation or result as displayed in the source.'
            }
        
        # Handle data queries with specific tone adaptation
        if content_type == 'data' or query_content_type == 'data':
            if question_tone == 'direct':
                return {
                    'instruction': 'Provide the specific data value requested with minimal context.',
                    'template': 'Answer: State the exact amount, percentage, or value'
                }
            else:
                return {
                    'instruction': 'Provide the data value with relevant context from the source material.',
                    'template': 'Answer: Include the specific value'
                }
        
        # Handle general content, adapting to question tone
        if question_tone == 'direct':
            return {
                'instruction': 'Provide straightforward answer focusing on the core question.',
                'template': 'Answer: Direct response based on source material.'
            }
        elif question_tone == 'complex':
            return {
                'instruction': 'Address the complexity of the question with comprehensive yet concise response.',
                'template': 'Answer: Thorough response covering key aspects from the source.'
            }
        else:
            return {
                'instruction': 'Provide clear, professional answer based strictly on source material.',
                'template': 'Answer: Professional response with relevant details from the source.'
            }    

    def _prepare_context(self, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """
        Prepares the context string from relevant document chunks for the LLM.
        Limits the context length to avoid exceeding model token limits.
        It now prioritizes shorter, high-scoring chunks to create a more diverse context.
        """
        context_parts = []
        total_length = 0
        max_context_for_detailed_answers = min(self.max_context_length, 3000)

        # Sort chunks by a combination of score (desc) and length (asc)
        # This gives priority to shorter, highly relevant chunks
        sorted_chunks = sorted(relevant_chunks, key=lambda x: (-x[1], len(x[0]['text'])))

        for chunk, score in sorted_chunks:
            chunk_text = chunk['text']
            chunk_length = len(chunk_text.split())

            if total_length + chunk_length > max_context_for_detailed_answers:
                continue # Skip chunks that would exceed the context limit

            enhanced_chunk = self._enhance_chunk_for_context(chunk_text)
            context_parts.append(enhanced_chunk)
            total_length += chunk_length

        return "\n\n".join(context_parts)

    def _enhance_chunk_for_context(self, chunk_text: str) -> str:
        """
        Cleans and enhances chunk text by removing escape characters and extra whitespace.
        """
        # Clean escape characters first
        chunk_text = clean_escape_characters(chunk_text)
        
        # Remove extra whitespace and clean formatting
        chunk_text = re.sub(r'\s+', ' ', chunk_text)
        chunk_text = chunk_text.strip()

        return chunk_text

    def _post_process_answer(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """
        Post-processes the raw answer from the LLM to clean it up and ensure consistency.
        Includes cleaning escape characters, fixing spacing, replacing quotes, and fixing URLs.
        """
        answer = answer.strip()

        # Clean escape characters
        answer = clean_escape_characters(answer)

        # Clean up spacing
        answer = re.sub(r'\s+', ' ', answer)

        # Replace double quotes with single quotes to avoid JSON escaping issues
        answer = answer.replace('"', "'")

        # Ensure the answer ends with a period for consistency
        if not answer.endswith('.'):
            answer += '.'

        return answer

    def _validate_and_enhance_answer(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> str:
        """
        Validates and enhances the completeness and specificity of the answer.
        (Currently not actively used in the main flow, but kept for potential future use).
        """
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
        """
        Enhances short answers by adding additional relevant details from the best chunk.
        """
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
        """
        Trims long answers while attempting to preserve essential information.
        """
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
        """
        Formats numbers in the answer for professional presentation.
        """
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
        """
        Generates a fallback answer when the primary LLM fails or no relevant information is found.
        """
        if not relevant_chunks:
            return "The provided document does not seem to contain information relevant to this question."

        # If the LLM fails, we can still try a simple, direct extraction as a last resort.
        best_chunk_text = relevant_chunks[0][0]['text']
        return f"While I couldn't formulate a definitive answer, the most relevant section I found states: \"{best_chunk_text[:250]}...\" This may provide some insight."

    def extract_reasoning(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> Dict:
        """
        Extracts reasoning and confidence metrics for a generated answer.
        """
        reasoning = {
            'confidence': self._calculate_confidence(answer, relevant_chunks),
            'source_chunks': [chunk['text'][:100] + "..." for chunk, _ in relevant_chunks[:3]],
            'reasoning': f"Answer derived from {len(relevant_chunks)} relevant policy sections"
        }

        return reasoning

    def _calculate_confidence(self, answer: str, relevant_chunks: List[Tuple[Dict, float]]) -> float:
        """
        Calculates a confidence score for the generated answer based on chunk relevance and answer indicators.
        """
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
