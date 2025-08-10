import json
import asyncio
import aiofiles
import os
from datetime import datetime
from typing import List
import logging

class APIRequestLogger:
    """
    Asynchronously logs incoming requests, including document URLs and questions,
    to a JSON file for auditing and analysis.
    """
    
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.session_file = None
        self._ensure_data_folder()
        self._create_session_file()
    
    def _ensure_data_folder(self):
        """
        Ensures that the data folder for logs exists. Creates it if it doesn't.
        """
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
    
    def _create_session_file(self):
        """
        Creates a new session file with a timestamped name.
        Initializes the file as an empty JSON array.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_file = os.path.join(self.data_folder, f"requests_session_{timestamp}.json")
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump([], f)
            logging.info(f"Created new request log session: {self.session_file}")
        except Exception as e:
            logging.error(f"Failed to create session file: {e}")
    
    async def log_request(self, document_url: str, questions: List[str]):
        """
        Logs a single request, appending document URL and questions to the current session file.
        """
        try:
            request_data = {
                "timestamp": datetime.now().isoformat(),
                "document_url": document_url,
                "questions": questions
            }
            
            await self._append_to_file(request_data)
            
        except Exception as e:
            logging.error(f"Failed to log request: {e}")
    
    async def _append_to_file(self, request_data: dict):
        """
        Appends the given request data to the JSON log file asynchronously.
        Reads existing data, appends new, and writes back.
        """
        try:
            # Read existing data from the file
            async with aiofiles.open(self.session_file, 'r') as f:
                content = await f.read()
                data = json.loads(content) if content.strip() else []
            
            # Append the new request data
            data.append(request_data)
            
            # Write the updated data back to the file
            async with aiofiles.open(self.session_file, 'w') as f:
                await f.write(json.dumps(data, indent=2))
                
        except Exception as e:
            logging.error(f"Error appending to log file: {e}")
    
    def get_session_file(self) -> str:
        """
        Returns the absolute path to the current session log file.
        """
        return self.session_file
