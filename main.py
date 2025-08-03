from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import asyncio
import logging

from src.document_processor import DocumentProcessor
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStore
from src.query_processor import QueryProcessor
from src.decision_engine import DecisionEngine
from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Process documents and answer queries with explainable decisions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answer: str
    confidence: Optional[float] = None
    source_chunks: Optional[List[str]] = None
    reasoning: Optional[str] = None

class QueryResponse(BaseModel):
    answers: List[str]

doc_processor = DocumentProcessor()
embedding_engine = EmbeddingEngine(Config.EMBEDDING_MODEL)
vector_store = VectorStore(embedding_engine.get_embedding_dimension())
query_processor = QueryProcessor(embedding_engine, vector_store)
decision_engine = DecisionEngine()

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    try:
        logging.info(f"Processing document: {request.documents}")
        logging.info(f"Number of questions: {len(request.questions)}")
        
        vector_store.clear()
        
        document_chunks = await doc_processor.process_document(request.documents)
        logging.info(f"Extracted {len(document_chunks)} chunks from document")
        
        embeddings = embedding_engine.generate_embeddings(document_chunks)
        vector_store.add_documents(document_chunks, embeddings)
        
        answers = []
        for i, question in enumerate(request.questions):
            logging.info(f"Processing question {i+1}/{len(request.questions)}: {question}")
            answer = await query_processor.process_query(
                question, 
                document_chunks,
                decision_engine
            )
            answers.append(answer)
        
        logging.info("All questions processed successfully")
        return QueryResponse(answers=answers)
    
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "query-retrieval-system"}

@app.post("/test")
async def test_endpoint():
    try:
        # Test with the hackathon document
        request = QueryRequest(
            documents="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            questions=["What is the grace period for premium payment?"]
        )
        
        vector_store.clear()
        document_chunks = await doc_processor.process_document(request.documents)
        embeddings = embedding_engine.generate_embeddings(document_chunks)
        vector_store.add_documents(document_chunks, embeddings)
        
        answer = await query_processor.process_query(
            request.questions[0], 
            document_chunks,
            decision_engine
        )
        
        return {"status": "success", "answer": answer}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)