from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
import time
import torch

from src.document_processor import DocumentProcessor
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStore
from src.query_processor import QueryProcessor
from src.decision_engine import DecisionEngine
from src.request_logger import RequestLogger
from src.file_validator import FileValidator, ValidationStatus
from config import Config

logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))

from contextlib import asynccontextmanager

# Create a global ThreadPoolExecutor

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Starting up...")
    app.state.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

    # Check for GPU availability and set device
    if torch.cuda.is_available():
        logging.info(f"CUDA GPU is available. Using device: {torch.cuda.get_device_name(0)}")
        app.state.device = "cuda"
        # Small test to confirm GPU is working
        try:
            a = torch.rand(1000, 1000).to(app.state.device)
            b = torch.rand(1000, 1000).to(app.state.device)
            _ = a @ b
            logging.info("GPU test successful: Tensor operations performed on GPU.")
        except Exception as e:
            logging.error(f"GPU test failed: {e}. Falling back to CPU.")
            app.state.device = "cpu"
    else:
        logging.info("CUDA GPU is not available. Models will use CPU.")
        app.state.device = "cpu"

    app.state.doc_processor = DocumentProcessor(app.state.executor)
    app.state.embedding_engine = EmbeddingEngine(Config.EMBEDDING_MODEL, app.state.device, app.state.executor)
    app.state.vector_store = VectorStore(app.state.embedding_engine.get_embedding_dimension(), app.state.executor)
    app.state.query_processor = QueryProcessor(app.state.embedding_engine, app.state.vector_store)
    app.state.decision_engine = DecisionEngine(device=app.state.device, executor=app.state.executor)
    app.state.request_logger = RequestLogger()
    app.state.file_validator = FileValidator()

    yield
    logging.info("Shutting down ThreadPoolExecutor...")
    app.state.executor.shutdown(wait=True)
    logging.info("ThreadPoolExecutor shut down.")

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Process documents and answer queries with explainable decisions",
    version="1.0.0",
    lifespan=lifespan
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





@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    request_start_time = time.time()
    try:
        # Log the request asynchronously (fire and forget)
        asyncio.create_task(app.state.request_logger.log_request(
            document_url=request.documents,
            questions=request.questions
        ))

        validation_status, message = await app.state.file_validator.validate_url(request.documents)

        if validation_status == ValidationStatus.UNSAFE:
            # Return a single answer for all questions, as the document is unsafe.
            return QueryResponse(answers=[message for _ in request.questions])
        
        if validation_status == ValidationStatus.ZIP_ARCHIVE:
            # Return a single answer with the contents of the zip file
            return QueryResponse(answers=[message for _ in request.questions])

        logging.info(f"Processing document: {request.documents}")
        logging.info(f"Number of questions: {len(request.questions)}")
        
        await app.state.vector_store.clear()
        
        document_chunks = await app.state.doc_processor.process_document(request.documents)
        logging.info(f"Extracted {len(document_chunks)} chunks from document")
        
        start_time_embeddings = time.time()
        embeddings = await app.state.embedding_engine.generate_embeddings(document_chunks)
        end_time_embeddings = time.time()
        logging.info(f"Embedding generation for {len(document_chunks)} chunks took {end_time_embeddings - start_time_embeddings:.2f} seconds.")
        await app.state.vector_store.add_documents(document_chunks, embeddings)
        
        answers = await app.state.query_processor.process_queries_parallel(
            request.questions, 
            document_chunks,
            app.state.decision_engine
        )
        
        request_end_time = time.time()
        total_request_time = round((request_end_time - request_start_time) * 1000, 2)
        logging.info(f"All questions processed successfully. Total request time: {total_request_time}ms")
        return QueryResponse(answers=answers)
    
    except Exception as e:
        request_end_time = time.time()
        total_request_time = round((request_end_time - request_start_time) * 1000, 2)
        logging.error(f"Error processing request: {str(e)} (Request time: {total_request_time}ms)")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "query-retrieval-system"}

@app.post("/test")
async def test_endpoint():
    test_start_time = time.time()
    try:
        logging.info("Starting test endpoint")
        # Test with the hackathon document
        request = QueryRequest(
            documents="https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
            questions=["What is the grace period for premium payment?"]
        )

        # Use app.state components
        await app.state.vector_store.clear()
        document_chunks = await app.state.doc_processor.process_document(request.documents)
        embeddings = await app.state.embedding_engine.generate_embeddings(document_chunks)
        await app.state.vector_store.add_documents(document_chunks, embeddings)

        answer = await app.state.query_processor.process_query(
            request.questions[0], 
            document_chunks,
            app.state.decision_engine
        )

        test_end_time = time.time()
        test_time = round((test_end_time - test_start_time) * 1000, 2)
        logging.info(f"Test endpoint completed in {test_time}ms")

        return {"status": "success", "answer": answer, "test_time_ms": test_time}

    except Exception as e:
        test_end_time = time.time()
        test_time = round((test_end_time - test_start_time) * 1000, 2)
        logging.error(f"Test endpoint failed: {str(e)} (Test time: {test_time}ms)")
        return {"status": "error", "message": str(e), "test_time_ms": test_time}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)