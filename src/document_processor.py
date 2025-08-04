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
import fitz
import os
from tempfile import NamedTemporaryFile
import functools
from concurrent.futures import ThreadPoolExecutor
import logging

class DocumentProcessor:
    def __init__(self, executor: ThreadPoolExecutor):
        self.executor = executor
        self.chunk_size = 512
        self.overlap = 50
        self._memory_cache = {}  # In-memory cache

    async def process_document(self, document_url: str) -> List[Dict]:
        logging.info(f"[DocumentProcessor] Started processing document: {document_url}")

        # Check memory cache
        if document_url in self._memory_cache:
            logging.info("[DocumentProcessor] Returning cached document from memory.")
            return self._memory_cache[document_url]

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            self.executor,
            functools.partial(self._process_document_sync, document_url)
        )

        logging.info("[DocumentProcessor] Finished parsing document")
        logging.info("[DocumentProcessor] Started chunking")
        chunks = self._create_chunks(text)
        logging.info(f"[DocumentProcessor] Finished chunking. Total chunks created: {len(chunks)}")

        # Save in memory cache
        self._memory_cache[document_url] = chunks

        return chunks

    def _process_document_sync(self, document_url: str) -> str:
        content_type = None
        try:
            head = requests.head(document_url)
            content_type = head.headers.get('content-type', '').lower()
        except Exception:
            logging.warning("[DocumentProcessor] HEAD request failed, skipping content-type check.")

        logging.info("[DocumentProcessor] Started parsing document")
        if 'pdf' in (content_type or '') or document_url.lower().endswith('.pdf'):
            text = self._extract_pdf_text_streamed(document_url)
        else:
            # fallback to full download for other formats
            response = requests.get(document_url)
            response.raise_for_status()

            if 'word' in (content_type or '') or document_url.lower().endswith(('.docx', '.doc')):
                text = self._extract_docx_text(BytesIO(response.content))
            elif 'email' in (content_type or '') or 'message' in (content_type or ''):
                text = self._extract_email_text(response.content)
            else:
                text = response.text
        return text

    def _extract_pdf_text_streamed(self, pdf_url: str) -> str:
        logging.info("[DocumentProcessor] Streaming and saving PDF temporarily...")

        with requests.get(pdf_url, stream=True) as r:
            r.raise_for_status()
            
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    temp_file.write(chunk)
                temp_path = temp_file.name

        logging.info("[DocumentProcessor] Parsing PDF from temp file...")
        text = ""
        try:
            with fitz.open(temp_path) as doc:
                for i, page in enumerate(doc):
                    text += page.get_text()
        finally:
            os.remove(temp_path)  # Clean up temp file

        return text

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