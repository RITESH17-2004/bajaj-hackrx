# HackRx Submission: LLM-Powered Intelligent Query-Retrieval System

## 🎯 Project Overview

This submission presents a comprehensive LLM-Powered Intelligent Query-Retrieval System designed specifically for processing insurance, legal, HR, and compliance documents. The system demonstrates advanced document understanding capabilities with explainable AI decisions.

## 🏗️ System Architecture

### Core Components Implemented:

1. **Document Processing Pipeline** (`src/document_processor.py`)
   - Multi-format support: PDF, DOCX, email documents
   - Intelligent chunking with overlap for context preservation
   - Text preprocessing and cleaning

2. **Embedding Engine** (`src/embedding_engine.py`)
   - Sentence-transformers integration (all-MiniLM-L6-v2)
   - Token-aware text truncation
   - Semantic similarity calculations

3. **Vector Store** (`src/vector_store.py`)
   - FAISS-powered similarity search
   - Normalized embeddings for optimal performance
   - Threshold-based relevance filtering

4. **Query Processor** (`src/query_processor.py`)
   - Intent classification (coverage, timing, information)
   - Entity extraction (medical terms, amounts, periods)
   - Hybrid search combining semantic and keyword matching

5. **Decision Engine** (`src/decision_engine.py`)
   - OpenAI GPT-3.5-turbo integration
   - Context-aware prompt engineering
   - Confidence scoring and explainability

6. **FastAPI Application** (`main.py`)
   - RESTful API matching exact specification
   - Comprehensive logging and error handling
   - CORS support for web integration

## 🚀 Key Features Implemented

### ✅ Document Processing
- **Multi-format Support**: PDF, DOCX, and email parsing
- **Intelligent Chunking**: 512-token chunks with 50-token overlap
- **Error Handling**: Robust exception handling for corrupted documents

### ✅ Semantic Search
- **Vector Embeddings**: High-quality sentence embeddings
- **FAISS Integration**: Efficient similarity search
- **Relevance Filtering**: Threshold-based result filtering

### ✅ Query Understanding
- **Intent Classification**: Automatically detects query type
- **Entity Extraction**: Identifies key terms, amounts, periods
- **Context Enhancement**: Keyword-based relevance boosting

### ✅ LLM Integration
- **OpenAI GPT Integration**: Professional-grade language model
- **Context-Aware Prompts**: Domain-specific prompt engineering
- **Token Optimization**: Efficient context management

### ✅ Explainable AI
- **Confidence Scoring**: Quantified answer reliability
- **Source Traceability**: Links answers to document sections
- **Reasoning Transparency**: Clear decision explanations

## 📊 Performance Optimization

### Token Efficiency
- **Smart Chunking**: Optimized context windows
- **Prompt Engineering**: Concise, effective prompts
- **Response Caching**: Reduced redundant API calls

### Speed Optimization
- **Async Processing**: Non-blocking document processing
- **Parallel Operations**: Concurrent embedding generation
- **Vector Indexing**: Fast similarity search

### Accuracy Enhancement
- **Hybrid Search**: Semantic + keyword matching
- **Re-ranking**: Relevance-based result ordering
- **Confidence Thresholding**: Quality assurance

## 🎯 Evaluation Criteria Addressed

### Accuracy ✅
- **Precise Query Understanding**: Intent classification and entity extraction
- **Semantic Matching**: High-quality embeddings and similarity search
- **Domain Expertise**: Insurance/legal-specific keyword recognition

### Token Efficiency ✅
- **Optimized Chunking**: Efficient context utilization
- **Smart Prompting**: Minimal token usage for maximum results
- **Caching Strategy**: Reduced API calls

### Latency ✅
- **Async Architecture**: Non-blocking operations
- **Vector Search**: Fast FAISS-based retrieval
- **Response Streaming**: Quick answer delivery

### Reusability ✅
- **Modular Design**: Independent, composable components
- **Configuration-Driven**: Easy customization
- **Docker Support**: Containerized deployment

### Explainability ✅
- **Source Attribution**: Clear document references
- **Confidence Metrics**: Quantified reliability
- **Decision Reasoning**: Transparent AI decisions

## 🔧 API Compliance

### Exact Specification Match
- **Endpoint**: `POST /hackrx/run`
- **Authentication**: Bearer token support
- **Request Format**: Compliant JSON structure
- **Response Format**: Exact array of strings as specified

### Additional Features
- **Health Check**: `GET /health` endpoint
- **API Documentation**: Automatic OpenAPI/Swagger docs
- **CORS Support**: Web application integration

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **LLM**: OpenAI GPT-3.5-turbo
- **Embeddings**: sentence-transformers
- **Vector DB**: FAISS
- **Document Processing**: PyPDF2, python-docx
- **Containerization**: Docker & Docker Compose

## 📋 Usage Instructions

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key-here"

# Start server
python start_server.py
```

### Docker Deployment
```bash
# Build and run
docker-compose up --build
```

### API Testing
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 250e6c57e9ef2aa5088d3bf610d72b73959b78486a62e066fd94ef74bc103c73" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?...",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## 🧪 Testing & Validation

### Component Testing
- All core components tested individually
- Import validation completed
- Error handling verified

### Integration Testing
- End-to-end API workflow tested
- Document processing pipeline validated
- Query-response cycle verified

### Performance Testing
- Server startup and response verified
- API endpoint accessibility confirmed
- Error handling robustness tested

## 📈 Scoring Optimization

### Document-Level Strategy
- **Known Documents**: Efficient processing with caching
- **Unknown Documents**: Robust parsing with fallbacks
- **Weighted Scoring**: Optimized for unknown document performance

### Question-Level Strategy
- **High-Weight Questions**: Priority processing
- **Complex Queries**: Enhanced context and reasoning
- **Edge Cases**: Graceful degradation with explanations

## 🚀 Deployment Ready

### Production Features
- **Environment Configuration**: Flexible deployment settings
- **Logging & Monitoring**: Comprehensive system observability
- **Error Handling**: Graceful failure management
- **Health Checks**: System status monitoring

### Scalability Considerations
- **Async Architecture**: Handles concurrent requests
- **Resource Management**: Efficient memory usage
- **Container Support**: Easy horizontal scaling

## 📝 Submission Checklist

✅ **Complete System Implementation**
✅ **API Specification Compliance**
✅ **Multi-format Document Processing**
✅ **Semantic Search with FAISS**
✅ **LLM Integration with Explainability**
✅ **Performance Optimization**
✅ **Comprehensive Documentation**
✅ **Testing & Validation**
✅ **Deployment Scripts**
✅ **Docker Configuration**

## 🏆 Innovation Highlights

1. **Hybrid Search**: Combines semantic embeddings with keyword matching
2. **Intent Classification**: Automatically understands query types
3. **Confidence Scoring**: Provides reliability metrics
4. **Domain Optimization**: Specialized for insurance/legal contexts
5. **Explainable AI**: Clear reasoning and source attribution

This submission represents a production-ready, highly optimized solution for the HackRx challenge, demonstrating both technical excellence and practical applicability in real-world document processing scenarios.