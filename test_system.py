import requests
import json
import asyncio
from src.document_processor import DocumentProcessor
from src.embedding_engine import EmbeddingEngine
from src.vector_store import VectorStore
from src.query_processor import QueryProcessor
from src.decision_engine import DecisionEngine

async def test_components():
    print("Testing individual components...")
    
    doc_processor = DocumentProcessor()
    embedding_engine = EmbeddingEngine()
    vector_store = VectorStore(embedding_engine.get_embedding_dimension())
    query_processor = QueryProcessor(embedding_engine, vector_store)
    decision_engine = DecisionEngine()
    
    sample_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    try:
        print("1. Testing document processing...")
        chunks = await doc_processor.process_document(sample_url)
        print(f"   ✓ Extracted {len(chunks)} chunks")
        
        print("2. Testing embedding generation...")
        embeddings = await embedding_engine.generate_embeddings(chunks[:5])
        print(f"   ✓ Generated embeddings shape: {embeddings.shape}")
        
        print("3. Testing vector store...")
        await vector_store.add_documents(chunks[:5], embeddings)
        document_count = await vector_store.get_document_count()
        print(f"   ✓ Added {document_count} documents to vector store")
        
        print("4. Testing query processing...")
        test_query = "What is the grace period for premium payment?"
        query_embedding = await embedding_engine.generate_query_embedding(test_query)
        results = await vector_store.search(query_embedding, k=3)
        print(f"   ✓ Found {len(results)} relevant chunks")
        
        print("All component tests passed!")
        
    except Exception as e:
        print(f"Component test failed: {str(e)}")

def test_api_endpoint():
    print("\nTesting API endpoint...")
    
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Authorization": "Bearer 250e6c57e9ef2aa5088d3bf610d72b73959b78486a62e066fd94ef74bc103c73",
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "Does this policy cover maternity expenses, and what are the conditions?"
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✓ API endpoint working!")
            print(f"Number of answers: {len(result['answers'])}")
            for i, answer in enumerate(result['answers']):
                print(f"Answer {i+1}: {answer[:100]}...")
        else:
            print(f"✗ API endpoint failed: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API (make sure server is running)")
    except Exception as e:
        print(f"✗ API test failed: {str(e)}")

if __name__ == "__main__":
    print("=== LLM Query-Retrieval System Test ===")
    
    asyncio.run(test_components())
    
    test_api_endpoint()