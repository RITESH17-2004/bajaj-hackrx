import re
import time
import logging
from typing import List, Dict, Tuple
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStore
from src.decision_engine import DecisionEngine

class QueryProcessor:
    def __init__(self, embedding_engine: EmbeddingEngine, vector_store: VectorStore):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.document_keywords = {
            'inclusion': ['include', 'included', 'contain', 'feature', 'cover'],
            'requirements': ['requirement', 'criteria', 'eligibility', 'condition'],
            'timing': ['period', 'time', 'duration', 'deadline', 'wait'],
            'amounts': ['amount', 'cost', 'price', 'fee', 'charge', 'rate'],
            'benefits': ['benefit', 'advantage', 'feature', 'service'],
            'exclusion': ['exclude', 'exclusion', 'not included', 'except'],
            'limits': ['limit', 'maximum', 'minimum', 'cap', 'ceiling'],
            'process': ['process', 'procedure', 'steps', 'method']
        }
    
    async def process_query(self, query: str, documents: List[Dict], decision_engine: DecisionEngine) -> str:
        start_time = time.time()
        logging.debug(f"Starting query processing: '{query[:50]}...'")
        
        # Step 1: Generate query embedding
        embedding_start = time.time()
        query_embedding = self.embedding_engine.generate_query_embedding(query)
        embedding_time = round((time.time() - embedding_start) * 1000, 2)
        logging.debug(f"Query embedding generated in {embedding_time}ms")
        
        # Step 2: Search for relevant chunks
        search_start = time.time()
        relevant_chunks = self.vector_store.search_with_threshold(
            query_embedding, 
            threshold=0.3, 
            k=10
        )
        
        if not relevant_chunks:
            relevant_chunks = self.vector_store.search(query_embedding, k=5)
        
        search_time = round((time.time() - search_start) * 1000, 2)
        logging.debug(f"Vector search completed in {search_time}ms, found {len(relevant_chunks)} chunks")
        
        # Step 3: Enhance chunks with keywords
        enhance_start = time.time()
        enhanced_chunks = self._enhance_with_keywords(query, relevant_chunks)
        enhance_time = round((time.time() - enhance_start) * 1000, 2)
        logging.debug(f"Chunk enhancement completed in {enhance_time}ms")
        
        # Step 4: Generate answer using decision engine
        answer_start = time.time()
        answer = await decision_engine.generate_answer(
            query, 
            enhanced_chunks, 
            self._extract_query_intent(query)
        )
        answer_time = round((time.time() - answer_start) * 1000, 2)
        logging.debug(f"Answer generation completed in {answer_time}ms")
        
        end_time = time.time()
        total_compute_time = round((end_time - start_time) * 1000, 2)
        
        # Log detailed timing breakdown (for monitoring only)
        logging.info(f"Query processing breakdown - Total: {total_compute_time}ms (Embedding: {embedding_time}ms, Search: {search_time}ms, Enhancement: {enhance_time}ms, Answer: {answer_time}ms)")
        
        # Return clean answer without timing
        return answer.strip()
    
    def _extract_query_intent(self, query: str) -> Dict:
        query_lower = query.lower()
        intent = {
            'type': 'general',
            'keywords': [],
            'entities': []
        }
        
        for category, keywords in self.document_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent['keywords'].append(category)
                    break
        
        if any(word in query_lower for word in ['does', 'is', 'are', 'include', 'contain']):
            intent['type'] = 'coverage'
        elif any(word in query_lower for word in ['what', 'how much', 'amount']):
            intent['type'] = 'information'
        elif any(word in query_lower for word in ['when', 'waiting', 'period']):
            intent['type'] = 'timing'
        
        entities = self._extract_entities(query)
        intent['entities'] = entities
        
        return intent
    
    def _extract_entities(self, query: str) -> List[str]:
        entities = []
        
        general_terms = re.findall(r'\b(?:process|procedure|requirement|condition|feature|service|benefit)\b', query.lower())
        entities.extend(general_terms)
        
        amounts = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', query)
        entities.extend(amounts)
        
        time_periods = re.findall(r'\b(?:\d+\s*(?:days?|months?|years?)|time\s+period)\b', query.lower())
        entities.extend(time_periods)
        
        return list(set(entities))
    
    def _enhance_with_keywords(self, query: str, chunks: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        query_words = set(query.lower().split())
        enhanced_chunks = []
        
        for chunk, score in chunks:
            chunk_words = set(chunk['text'].lower().split())
            keyword_overlap = len(query_words.intersection(chunk_words))
            
            keyword_boost = min(keyword_overlap * 0.1, 0.5)
            enhanced_score = score + keyword_boost
            
            enhanced_chunks.append((chunk, enhanced_score))
        
        enhanced_chunks.sort(key=lambda x: x[1], reverse=True)
        return enhanced_chunks
    
    def _calculate_relevance_score(self, query: str, chunk: Dict) -> float:
        query_lower = query.lower()
        chunk_text_lower = chunk['text'].lower()
        
        exact_matches = sum(1 for word in query.split() if word.lower() in chunk_text_lower)
        total_query_words = len(query.split())
        
        relevance_score = exact_matches / total_query_words if total_query_words > 0 else 0
        
        for category, keywords in self.document_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if any(keyword in chunk_text_lower for keyword in keywords):
                    relevance_score += 0.2
        
        return min(relevance_score, 1.0)