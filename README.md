<<<<<<< HEAD
# LLM-Powered Intelligent Query-Retrieval System

An advanced document processing and query answering system designed for insurance, legal, HR, and compliance domains. The system processes PDFs, DOCX, and email documents to provide accurate, contextual answers with explainable reasoning.

## 🚀 Features

- **Multi-format Document Processing**: PDF, DOCX, and email support
- **Semantic Search**: FAISS-powered vector similarity search
- **LLM Integration**: OpenAI GPT-4/3.5-turbo for intelligent answers
- **Explainable Decisions**: Confidence scores and source traceability
- **Token Optimization**: Efficient prompt engineering and caching
- **RESTful API**: FastAPI-based with automatic documentation

## 🏗️ System Architecture

```
Document URL → Document Parser → Text Chunks → Embeddings → Vector Store
Query → Query Parser → Embedding Search → Clause Matching → Decision Logic → JSON Response
```

### Core Components:
1. **Document Processor** - Extracts text from various document formats
2. **Embedding Engine** - Generates semantic embeddings using sentence-transformers
3. **Vector Store** - FAISS-based similarity search
4. **Query Processor** - Natural language understanding and context matching
5. **Decision Engine** - LLM-powered answer generation with explainability

## 📋 Requirements

- Python 3.8+
- OpenAI API Key (for GPT models)
- 4GB+ RAM (for embedding models)
- Docker (optional)

## 🛠️ Installation

### Option 1: Local Setup

1. **Clone and navigate to the project:**
   ```bash
   cd bajaj
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Start the server:**
   ```bash
   python start_server.py
   ```

### Option 2: Docker Setup

1. **Build and run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

## 🔧 Configuration

Configure the system by editing `.env` or setting environment variables:

```env
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
MODEL_CACHE_DIR=./models
LOG_LEVEL=INFO
MAX_WORKERS=4
```

## 📚 API Usage

### Main Endpoint

**POST** `/hackrx/run`

**Headers:**
```
Authorization: Bearer 250e6c57e9ef2aa5088d3bf610d72b73959b78486a62e066fd94ef74bc103c73
Content-Type: application/json
```

**Request Body:**
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "Does this policy cover maternity expenses?"
    ]
}
```

**Response:**
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date...",
        "Yes, the policy covers maternity expenses with a 24-month waiting period..."
    ]
}
```

### Health Check

**GET** `/health`

Returns system status and health information.

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## 🧪 Testing

### Run Component Tests
```bash
python test_system.py
```

### Test API Endpoint
```bash
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer 250e6c57e9ef2aa5088d3bf610d72b73959b78486a62e066fd94ef74bc103c73" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
      "What is the grace period for premium payment?",
      "Does this policy cover maternity expenses?"
    ]
  }'
```

## 🎯 Performance Optimization

### Token Efficiency
- Chunked document processing (512 tokens max)
- Optimized prompts for concise responses
- Caching for frequently accessed documents

### Speed Optimization
- Async processing pipeline
- Parallel embedding generation
- Vector index optimization

### Accuracy Enhancement
- Hybrid search (semantic + keyword matching)
- Query intent classification
- Confidence scoring and re-ranking

## 📊 Evaluation Metrics

The system is designed to optimize:
- **Accuracy**: Query understanding and clause matching precision
- **Token Efficiency**: Cost-effective LLM usage
- **Latency**: <2s response time target
- **Explainability**: Clear reasoning with source traceability

## 🔍 Domain-Specific Features

### Insurance Documents
- Policy coverage analysis
- Waiting period extraction
- Premium and claim processing
- Exclusion identification

### Legal Documents
- Contract clause analysis
- Liability assessment
- Compliance checking
- Risk evaluation

### HR & Compliance
- Policy interpretation
- Benefit analysis
- Regulatory compliance
- Procedure documentation

## 🐛 Troubleshooting

### Common Issues

1. **Model Download Issues**
   ```bash
   # Manually download models
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

2. **Memory Issues**
   - Reduce chunk size in `config.py`
   - Use smaller embedding models
   - Implement document pagination

3. **API Key Issues**
   - Verify API keys in `.env`
   - Check API quota and billing

4. **Connection Issues**
   - Verify document URLs are accessible
   - Check firewall and network settings

## 📈 Scaling Considerations

For production deployment:
- Use Pinecone for scalable vector storage
- Implement connection pooling
- Add caching layer (Redis)
- Use container orchestration (Kubernetes)
- Monitor with OpenTelemetry

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project is developed for the HackRx hackathon and is provided as-is for evaluation purposes.

## 🔗 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [FAISS Documentation](https://faiss.ai/)
- [OpenAI API Reference](https://platform.openai.com/docs)
=======
# bajaj-finserv
>>>>>>> 44caf1b56870c5b376be14bc04d2942166954374
