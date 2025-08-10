# -*- coding: utf-8 -*-
"""
Main FastAPI application file for the LLM-Powered Intelligent Query-Retrieval System.
This file sets up the FastAPI application, defines the API endpoints, and handles the
core logic for processing documents and answering queries.
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import asyncio
import logging
import time
import torch
import os
import hashlib

from src.document_text_extractor import DocumentTextExtractor
from src.embedding_generator import EmbeddingGenerator
from src.faiss_vector_store import FAISSVectorStore
from src.query_resolver import QueryResolver
from src.answer_generation_engine import AnswerGenerationEngine
from src.api_request_logger import APIRequestLogger
from src.input_validator import InputValidator, ValidationStatus
from config import Config
from src.intelligent_agent import IntelligentAgent
from src.llm_interaction_service import LLMInteractionService
from src.open_source_llm_engine import OpenSourceLLMEngine

# Configure logging for the application
logging.basicConfig(level=getattr(logging, Config.LOG_LEVEL))

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown.
    Initializes core components like thread pool, document processor, embedding engine,
    vector store, query processor, decision engine, request logger, and file validator.
    Checks for GPU availability and sets the device for torch operations.
    """
    logging.info("Application starting up...")
    app.state.executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)

    # Create cache directory for vector stores
    app.state.cache_dir = ".cache/vector_stores"
    os.makedirs(app.state.cache_dir, exist_ok=True)

    # Determine and set the appropriate device (GPU/CPU) for torch
    if torch.cuda.is_available():
        logging.info(f"CUDA GPU available: {torch.cuda.get_device_name(0)}. Using CUDA.")
        app.state.device = "cuda"
        try:
            # Test GPU functionality
            _ = torch.rand(1000, 1000).to(app.state.device) @ torch.rand(1000, 1000).to(app.state.device)
            logging.info("GPU test successful.")
        except Exception as e:
            logging.error(f"GPU test failed: {e}. Falling back to CPU.")
            app.state.device = "cpu"
    else:
        logging.info("No CUDA GPU found. Using CPU.")
        app.state.device = "cpu"

    # Initialize core application components
    app.state.doc_processor = DocumentTextExtractor(app.state.executor)
    app.state.embedding_engine = EmbeddingGenerator(Config.EMBEDDING_MODEL, app.state.device, app.state.executor)
    app.state.vector_store = FAISSVectorStore(app.state.embedding_engine.get_embedding_dimension(), app.state.executor)
    app.state.query_processor = QueryResolver(app.state.embedding_engine, app.state.vector_store)
    app.state.decision_engine = AnswerGenerationEngine(device=app.state.device, executor=app.state.executor)
    app.state.request_logger = APIRequestLogger()
    app.state.file_validator = InputValidator()

    yield
    
    # Gracefully shut down application components
    logging.info("Application shutting down...")
    await app.state.doc_processor.shutdown()
    app.state.executor.shutdown(wait=True)
    logging.info("Application shutdown complete.")

# Initialize FastAPI application
app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="Process documents and answer queries with explainable decisions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup HTTP Bearer token security
security = HTTPBearer()

class QueryRequest(BaseModel):
    """Request model for document and question submission."""
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    """Response model for answers to submitted questions."""
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Processes a document from a URL and answers a list of questions.
    Authenticates requests using a bearer token.
    Includes document validation, caching, embedding generation, and dynamic routing to an agent or RAG pipeline.
    """
    # Authenticate the request with bearer token
    if not credentials or credentials.scheme.lower() != "bearer" or credentials.credentials != Config.BEARER_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
    
    request_start_time = time.time()
    try:
        # Asynchronously log the incoming request details
        asyncio.create_task(app.state.request_logger.log_request(
            document_url=request.documents,
            questions=request.questions
        ))

        # Validate the provided document URL for safety and type
        validation_status, message = await app.state.file_validator.validate_url(request.documents)
        if validation_status in (ValidationStatus.UNSAFE, ValidationStatus.ZIP_ARCHIVE):
            return QueryResponse(answers=[message for _ in request.questions])

        # Generate a unique cache filename from the document URL
        doc_url = request.documents
        cache_filename = hashlib.sha256(doc_url.encode('utf-8')).hexdigest()
        cache_filepath = os.path.join(app.state.cache_dir, cache_filename)

        # Clear previous vector store data and attempt to load from cache
        await app.state.vector_store.clear()
        is_cached, full_document_text = await app.state.vector_store.load_index(cache_filepath)

        if is_cached:
            logging.info(f"Cache HIT for document: {doc_url}")
            document_chunks = app.state.vector_store.documents
        else:
            logging.info(f"Cache MISS for document: {doc_url}. Processing...")
            # Process document to extract chunks and full text
            document_chunks = [chunk async for chunk in app.state.doc_processor.process_document(doc_url)]
            full_document_text = " ".join([chunk['text'] for chunk in document_chunks])

            # Generate embeddings for document chunks and add to vector store in batches
            chunk_batch = []
            batch_size = 32
            start_time_embeddings = time.time()
            
            for chunk in document_chunks:
                chunk_batch.append(chunk)
                if len(chunk_batch) >= batch_size:
                    embeddings = await app.state.embedding_engine.generate_embeddings(chunk_batch)
                    await app.state.vector_store.add_documents(chunk_batch, embeddings)
                    chunk_batch = []
            
            if chunk_batch:
                embeddings = await app.state.embedding_engine.generate_embeddings(chunk_batch)
                await app.state.vector_store.add_documents(chunk_batch, embeddings)

            logging.info(f"Embedding generation for {len(document_chunks)} chunks took {time.time() - start_time_embeddings:.2f} seconds.")

            # Save the newly processed index to cache
            await app.state.vector_store.save_index(cache_filepath, full_document_text)
            logging.info(f"Finished processing and cached vector store for document: {doc_url}")

        # Determine if a dynamic agent or standard RAG pipeline is required
        if app.state.file_validator.is_agent_required(full_document_text):
            logging.info("Dynamic document detected. Routing to Mission Agent.")
            
            # Initialize LLM engine for the agent based on API key availability
            mistral_api_key = os.getenv('MISTRAL_API_KEY')
            if not (mistral_api_key and mistral_api_key != 'your-mistral-api-key-here'):
                raise HTTPException(status_code=503, detail="Mistral API key not configured. Mistral LLM is required for dynamic documents.")
            
            logging.info("Using Mistral LLM Engine for agent.")
            from src.mistral_api_llm_engine import MistralApiLLMEngine
            llm_engine_for_agent = MistralApiLLMEngine(app.state.executor)

            # Run the intelligent agent to get answers
            llm_service = LLMInteractionService(llm_engine_for_agent)
            intelligent_agent = IntelligentAgent(llm_service, app.state.query_processor, app.state.decision_engine)
            answer = await intelligent_agent.run(document_chunks, full_document_text)
            answers = [answer] * len(request.questions)
        
        else:
            logging.info("Standard document detected. Using RAG pipeline.")
            # Process queries using the standard RAG pipeline
            answers = await app.state.query_processor.process_queries_parallel(
                request.questions, 
                document_chunks,
                app.state.decision_engine
            )
        
        total_request_time = round((time.time() - request_start_time) * 1000, 2)
        logging.info(f"Request processed successfully in {total_request_time}ms")
        return QueryResponse(answers=answers)
    
    except Exception as e:
        total_request_time = round((time.time() - request_start_time) * 1000, 2)
        logging.error(f"Error processing request: {str(e)} (Request time: {total_request_time}ms)")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.get("/health")
async def health_check():
    """Returns a health status to indicate the service is running."""
    return {"status": "healthy", "service": "query-retrieval-system"}

if __name__ == "__main__":
    # Run the FastAPI application using uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=8000)
