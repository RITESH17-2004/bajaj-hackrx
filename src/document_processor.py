import requests
import PyPDF2
import docx
import email
import re
import os
import concurrent.futures
from typing import List, Dict, Optional
from io import BytesIO
from bs4 import BeautifulSoup
import aiofiles
import asyncio
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    def __init__(self):
        # Default values - will be adjusted based on document size
        self.chunk_size = 512
        self.overlap = 50
        self.estimated_words_per_page = 500  # Average words per page
        
        # Advanced chunking settings from env
        self.adaptive_chunking = os.getenv('ADAPTIVE_CHUNKING', 'true').lower() == 'true'
        self.smart_overlap = os.getenv('SMART_OVERLAP', 'true').lower() == 'true'
        self.parallel_processing = os.getenv('PARALLEL_PROCESSING', 'true').lower() == 'true'
        
        # Initialize semantic chunking model (lightweight for speed)
        self.semantic_model = None
        if self.adaptive_chunking:
            try:
                self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                print("Semantic chunking enabled with all-MiniLM-L6-v2")
            except Exception as e:
                print(f"Failed to load semantic model: {e}, falling back to rule-based chunking")
    
    async def process_document(self, document_url: str) -> List[Dict]:
        response = requests.get(document_url)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        
        if 'pdf' in content_type or document_url.lower().endswith('.pdf'):
            text = self._extract_pdf_text(BytesIO(response.content))
        elif 'word' in content_type or document_url.lower().endswith(('.docx', '.doc')):
            text = self._extract_docx_text(BytesIO(response.content))
        elif 'email' in content_type or 'message' in content_type:
            text = self._extract_email_text(response.content)
        else:
            text = response.text
        
        # Estimate document size and adjust chunking strategy
        page_count = self._estimate_page_count(text)
        self._adjust_chunking_strategy(page_count)
        
        # Use faster chunking strategy based on document size
        if page_count > 100:
            print("Using fast chunking for large document")
            chunks = self._create_fast_chunks(text, page_count)
        else:
            chunks = self._create_chunks(text, page_count)
        
        return chunks
    
    def _extract_pdf_text(self, pdf_bytes: BytesIO) -> str:
        try:
            reader = PyPDF2.PdfReader(pdf_bytes)
            text = ""
            page_count = len(reader.pages)
            print(f"Processing PDF with {page_count} pages")
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def _extract_docx_text(self, docx_bytes: BytesIO) -> str:
        try:
            doc = docx.Document(docx_bytes)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {str(e)}")
    
    def _extract_email_text(self, email_content: bytes) -> str:
        try:
            msg = email.message_from_bytes(email_content)
            text = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    elif part.get_content_type() == "text/html":
                        html_content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                        soup = BeautifulSoup(html_content, 'html.parser')
                        text += soup.get_text()
            else:
                if msg.get_content_type() == "text/plain":
                    text = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                elif msg.get_content_type() == "text/html":
                    html_content = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(html_content, 'html.parser')
                    text = soup.get_text()
            
            return text
        except Exception as e:
            raise Exception(f"Error extracting email text: {str(e)}")
    
    def _estimate_page_count(self, text: str) -> int:
        """Estimate page count based on word count"""
        word_count = len(text.split())
        estimated_pages = max(1, word_count // self.estimated_words_per_page)
        print(f"Estimated page count: {estimated_pages} pages ({word_count} words)")
        return estimated_pages
    
    def _adjust_chunking_strategy(self, page_count: int):
        """Adjust chunking parameters based on document size"""
        print(f"Adjusting chunking strategy for {page_count} pages")
        
        if page_count <= 1:
            # Small documents (1 page): Fine-grained chunks
            self.chunk_size = 256
            self.overlap = 32
            print("Strategy: Fine-grained chunking")
            
        elif page_count <= 50:
            # Medium documents (2-50 pages): Standard chunks
            self.chunk_size = 512
            self.overlap = 64
            print("Strategy: Standard chunking")
            
        elif page_count <= 100:
            # Large documents (51-100 pages): Larger chunks for efficiency
            self.chunk_size = 768
            self.overlap = 96
            print("Strategy: Large chunk processing")
            
        elif page_count <= 500:
            # Very large documents (101-500 pages): High-efficiency chunks
            self.chunk_size = 1024
            self.overlap = 128
            print("Strategy: High-efficiency processing")
            
        else:
            # Massive documents (500+ pages): Maximum efficiency
            self.chunk_size = 1536
            self.overlap = 192
            print("Strategy: Maximum efficiency processing")
        
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.overlap}")
    
    def _create_chunks(self, text: str, page_count: int = None) -> List[Dict]:
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'chunk_id': len(chunks),
                    'token_count': current_length,
                    'page_count': page_count,
                    'chunk_strategy': self._get_strategy_name(page_count)
                })
                
                if len(current_chunk.split()) > self.overlap:
                    overlap_words = current_chunk.split()[-self.overlap:]
                    current_chunk = ' '.join(overlap_words) + ' ' + sentence
                    current_length = len(overlap_words) + sentence_length
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
                current_length += sentence_length
        
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'chunk_id': len(chunks),
                'token_count': current_length,
                'page_count': page_count,
                'chunk_strategy': self._get_strategy_name(page_count)
            })
        
        print(f"Created {len(chunks)} chunks using {self._get_strategy_name(page_count)} strategy")
        return chunks
    
    def _create_fast_chunks(self, text: str, page_count: int) -> List[Dict]:
        """Fast chunking for large documents - optimized for speed"""
        print("Using fast chunking algorithm")
        
        # Simple word-based chunking for speed
        words = text.split()
        chunks = []
        chunk_id = 0
        
        # Calculate words per chunk based on chunk_size
        words_per_chunk = self.chunk_size
        overlap_words = self.overlap
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk_words = words[i:i + words_per_chunk]
            
            if not chunk_words:
                break
                
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'chunk_id': chunk_id,
                'token_count': len(chunk_words),
                'page_count': page_count,
                'chunk_strategy': 'fast-chunking'
            })
            
            chunk_id += 1
            
            # Stop if we've reached the end
            if i + words_per_chunk >= len(words):
                break
        
        print(f"Created {len(chunks)} chunks using fast chunking")
        return chunks
    
    def _get_strategy_name(self, page_count: int) -> str:
        """Get the strategy name based on page count"""
        if page_count <= 1:
            return "fine-grained"
        elif page_count <= 50:
            return "standard"
        elif page_count <= 100:
            return "large-chunks"
        elif page_count <= 500:
            return "high-efficiency"
        else:
            return "maximum-efficiency"
    
    async def _create_semantic_chunks(self, text: str, page_count: int) -> List[Dict]:
        """Advanced semantic chunking using sentence embeddings"""
        print("Starting semantic chunking process...")
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        print(f"Split text into {len(sentences)} sentences")
        
        if len(sentences) < 10:
            # Too few sentences, use regular chunking
            return self._create_chunks(text, page_count)
        
        # Use parallel processing for large documents
        if self.parallel_processing and len(sentences) > 100:
            return await self._parallel_semantic_chunking(sentences, page_count)
        else:
            return await self._sequential_semantic_chunking(sentences, page_count)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with better handling"""
        # Enhanced sentence splitting
        sentence_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence endings
            r'(?<=[.!?])\s*\n+\s*(?=[A-Z])',  # Sentence endings with newlines
            r'(?<=\w\.)\s+(?=[A-Z][a-z])',  # Handle abbreviations better
        ]
        
        sentences = [text]
        for pattern in sentence_patterns:
            new_sentences = []
            for sentence in sentences:
                new_sentences.extend(re.split(pattern, sentence))
            sentences = new_sentences
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence.split()) > 3:  # Filter out very short sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    async def _parallel_semantic_chunking(self, sentences: List[str], page_count: int) -> List[Dict]:
        """Parallel processing for semantic chunking of large documents"""
        print("Using parallel semantic chunking")
        
        # Split sentences into batches for parallel processing
        batch_size = max(50, len(sentences) // os.cpu_count())
        sentence_batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
        
        # Process batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, os.cpu_count())) as executor:
            batch_results = list(executor.map(self._process_sentence_batch, sentence_batches))
        
        # Combine results
        all_chunks = []
        chunk_id = 0
        for batch_chunks in batch_results:
            for chunk in batch_chunks:
                chunk['chunk_id'] = chunk_id
                chunk['page_count'] = page_count
                chunk['chunk_strategy'] = 'semantic-parallel'
                all_chunks.append(chunk)
                chunk_id += 1
        
        print(f"Created {len(all_chunks)} chunks using parallel semantic chunking")
        return all_chunks
    
    def _process_sentence_batch(self, sentence_batch: List[str]) -> List[Dict]:
        """Process a batch of sentences for semantic chunking"""
        if not sentence_batch:
            return []
        
        try:
            # Generate embeddings for the batch
            embeddings = self.semantic_model.encode(sentence_batch)
            
            # Calculate semantic similarity between adjacent sentences
            chunks = []
            current_chunk = [sentence_batch[0]]
            current_embedding = [embeddings[0]]
            
            similarity_threshold = 0.75  # Adjusted for better chunking
            
            for i in range(1, len(sentence_batch)):
                # Calculate similarity with current chunk
                chunk_avg_embedding = np.mean(current_embedding, axis=0)
                similarity = np.dot(chunk_avg_embedding, embeddings[i]) / (
                    np.linalg.norm(chunk_avg_embedding) * np.linalg.norm(embeddings[i])
                )
                
                # Check if we should start a new chunk
                current_chunk_size = sum(len(s.split()) for s in current_chunk)
                
                if (similarity < similarity_threshold or 
                    current_chunk_size > self.chunk_size):
                    
                    # Create chunk from current sentences
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'token_count': len(chunk_text.split()),
                        'semantic_similarity': float(similarity)
                    })
                    
                    # Start new chunk
                    current_chunk = [sentence_batch[i]]
                    current_embedding = [embeddings[i]]
                else:
                    # Add to current chunk
                    current_chunk.append(sentence_batch[i])
                    current_embedding.append(embeddings[i])
            
            # Add final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'token_count': len(chunk_text.split()),
                    'semantic_similarity': 1.0
                })
            
            return chunks
            
        except Exception as e:
            print(f"Error in semantic chunking batch: {e}")
            # Fallback to simple chunking for this batch
            return [{'text': ' '.join(sentence_batch), 'token_count': len(' '.join(sentence_batch).split())}]
    
    async def _sequential_semantic_chunking(self, sentences: List[str], page_count: int) -> List[Dict]:
        """Sequential semantic chunking for smaller documents"""
        print("Using sequential semantic chunking")
        
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 20
            all_chunks = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                batch_chunks = self._process_sentence_batch(batch)
                all_chunks.extend(batch_chunks)
            
            # Add metadata
            for i, chunk in enumerate(all_chunks):
                chunk['chunk_id'] = i
                chunk['page_count'] = page_count
                chunk['chunk_strategy'] = 'semantic-sequential'
            
            print(f"Created {len(all_chunks)} chunks using sequential semantic chunking")
            return all_chunks
            
        except Exception as e:
            print(f"Error in semantic chunking: {e}, falling back to rule-based")
            return self._create_chunks(' '.join(sentences), page_count)
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()