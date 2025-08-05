import json
import asyncio
import aiofiles
import os
from datetime import datetime
from typing import List
import logging

class RequestLogger:
    """Simple async logger for storing document URLs and questions"""
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.session_file = None
        self._ensure_data_folder()
        self._create_session_file()
    
    def _ensure_data_folder(self):
        """Ensure data folder exists"""
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
    
    def _create_session_file(self):
        """Create new session file based on current timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(self.data_folder, f"requests_session_{timestamp}.json")
        
        # Initialize empty JSON array
        try:
            with open(self.session_file, 'w') as f:
                json.dump([], f)
            logging.info(f"Created new request log session: {self.session_file}")
        except Exception as e:
            logging.error(f"Failed to create session file: {e}")
    
    async def log_request(self, document_url: str, questions: List[str]):
        """Log a request with document URL and questions"""
        try:
            request_data = {
                "timestamp": datetime.now().isoformat(),
                "document_url": document_url,
                "questions": questions
            }
            
            # Read current data, append new request, write back
            await self._append_to_file(request_data)
            
        except Exception as e:
            logging.error(f"Failed to log request: {e}")
    
    async def _append_to_file(self, request_data: dict):
        """Append request data to JSON file asynchronously"""
        try:
            # Read existing data
            async with aiofiles.open(self.session_file, 'r') as f:
                content = await f.read()
                data = json.loads(content) if content.strip() else []
            
            # Append new request
            data.append(request_data)
            
            # Write back to file
            async with aiofiles.open(self.session_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logging.error(f"Error appending to log file: {e}")
    
    def get_session_file(self) -> str:
        """Get current session file path"""
        return self.session_file