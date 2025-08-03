import requests
import PyPDF2
import docx
import email
import re
from typing import List, Dict
from io import BytesIO
from bs4 import BeautifulSoup
import aiofiles
import asyncio

class DocumentProcessor:
    def __init__(self):
        self.chunk_size = 512
        self.overlap = 50
    
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
        
        chunks = self._create_chunks(text)
        return chunks
    
    def _extract_pdf_text(self, pdf_bytes: BytesIO) -> str:
        try:
            reader = PyPDF2.PdfReader(pdf_bytes)
            text = ""
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
    
    def _create_chunks(self, text: str) -> List[Dict]:
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
                    'token_count': current_length
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
                'token_count': current_length
            })
        
        return chunks
    
    def preprocess_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\-\(\)]', '', text)
        return text.strip()