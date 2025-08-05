import asyncio
import re
import logging
from typing import List, Dict, Tuple
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStore
from src.decision_engine import DecisionEngine
from config import Config

class QueryProcessor:
    def __init__(self, embedding_engine: EmbeddingEngine, vector_store: VectorStore, concurrency_limit: int = Config.MAX_WORKERS):
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.insurance_keywords = {
            'coverage': ['cover', 'coverage', 'covered', 'include', 'included'],
            'conditions': ['condition', 'requirement', 'criteria', 'eligibility'],
            'waiting_period': ['waiting period', 'wait', 'waiting time'],
            'premium': ['premium', 'payment', 'cost', 'price'],
            'claim': ['claim', 'benefit', 'reimbursement'],
            'exclusion': ['exclude', 'exclusion', 'not covered', 'except'],
            'limit': ['limit', 'maximum', 'cap', 'ceiling'],
            'deductible': ['deductible', 'excess', 'co-pay']
        }

    async def process_query(self, query: str, documents: List[Dict], decision_engine: DecisionEngine) -> str:
        async with self.semaphore:
            logging.info(f"Processing query: {query}")
            
            query_embedding = await self.embedding_engine.generate_query_embedding(query)
            logging.info(f"Generated query embedding. Type: {type(query_embedding)}")
            
            relevant_chunks = await self.vector_store.search(query_embedding, k=15)
            logging.info(f"Found {len(relevant_chunks)} relevant chunks.")
            
            enhanced_chunks = self._enhance_with_keywords(query, relevant_chunks)
            logging.info("Enhanced chunks with keywords.")
            
            query_intent = self._extract_query_intent(query)
            logging.info(f"Extracted query intent: {query_intent}")
            
            answer = await decision_engine.generate_answer(
                query, 
                enhanced_chunks, 
                query_intent
            )
            logging.info(f"Generated answer. Type: {type(answer)}")
            
            return answer

    async def process_queries_parallel(self, queries: List[str], documents: List[Dict], decision_engine: DecisionEngine) -> List[str]:
        tasks = [self.process_query(query, documents, decision_engine) for query in queries]
        return await asyncio.gather(*tasks)

    def _extract_query_intent(self, query: str) -> Dict:
        query_lower = query.lower()
        intent = {
            'type': 'general',
            'keywords': [],
            'entities': []
        }
        
        for category, keywords in self.insurance_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent['keywords'].append(category)
                    break
        
        if any(word in query_lower for word in ['does', 'is', 'are', 'cover', 'include']):
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
        
        medical_terms = re.findall(r'\b(?:surgery|treatment|therapy|procedure|condition|disease|illness|injury)\b', query.lower())
        entities.extend(medical_terms)
        
        amounts = re.findall(r'\$?\d+(?:,\d{3})*(?:\.\d{2})?', query)
        entities.extend(amounts)
        
        time_periods = re.findall(r'\b(?:\d+\s*(?:days?|months?|years?)|waiting\s+period)\b', query.lower())
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
        
        for category, keywords in self.insurance_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                if any(keyword in chunk_text_lower for keyword in keywords):
                    relevance_score += 0.2
        
        return min(relevance_score, 1.0)