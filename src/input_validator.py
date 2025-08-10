import requests
import zipfile
from io import BytesIO
import logging
from typing import Tuple, Optional, List
from enum import Enum
import asyncio
import re

class ValidationStatus(Enum):
    """Enum to represent the validation status of a document URL."""
    SAFE_TO_PROCESS = 1
    UNSAFE = 2
    ZIP_ARCHIVE = 3

class InputValidator:
    """
    Validates document URLs for safety and type, especially handling zip archives.
    Also determines if a dynamic agent is required based on document content.
    """
    def __init__(self, max_zip_size=100 * 1024 * 1024, max_uncompressed_size=500 * 1024 * 1024, max_files=1000):
        self.max_zip_size = max_zip_size              # Maximum allowed size for a zip file (100 MB)
        self.max_uncompressed_size = max_uncompressed_size # Maximum allowed uncompressed size of zip contents (500 MB)
        self.max_files = max_files                    # Maximum allowed number of files within a zip archive

    def is_agent_required(self, document_text: str) -> bool:
        """
        Checks if the document text contains patterns that suggest an API endpoint URL,
        indicating that a dynamic agent might be required for processing.
        """
        # Check for the presence of what looks like an API endpoint URL
        if re.search(r"https?://[^/]+/.+", document_text):
            return True
        return False

    async def validate_url(self, document_url: str) -> Tuple[ValidationStatus, Optional[str]]:
        """
        Validates the given document URL.
        Checks for binary files, handles zip archives, and returns a validation status.
        """
        if document_url.lower().endswith('.bin'):
            return ValidationStatus.UNSAFE, "The provided URL points to a binary file, not a readable document."

        if document_url.lower().endswith('.zip'):
            status, message = await self._handle_zip(document_url)
            if status: # If zip is safe to process
                return ValidationStatus.ZIP_ARCHIVE, message
            else: # If zip is unsafe
                return ValidationStatus.UNSAFE, message

        return ValidationStatus.SAFE_TO_PROCESS, None

    async def _handle_zip(self, zip_url: str) -> Tuple[bool, Optional[str]]:
        """
        Asynchronously handles the validation of a zip file from a URL.
        Checks size limits and file counts within the zip.
        """
        try:
            # Run the synchronous zip handling in a separate thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_handle_zip, zip_url)
        except Exception as e:
            logging.error(f"Unexpected error handling zip file: {e}")
            return False, "An unexpected error occurred while processing the zip file."

    def _sync_handle_zip(self, zip_url: str) -> Tuple[bool, Optional[str]]:
        """
        Synchronously downloads and inspects a zip file for safety.
        Checks against defined size and file count limits.
        """
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()

            # Check content-length header for initial size validation
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_zip_size:
                return False, "The provided zip file is too large to process."

            # Download zip content in chunks and check size during download
            zip_content = BytesIO()
            size = 0
            for chunk in response.iter_content(1024*1024):
                size += len(chunk)
                if size > self.max_zip_size:
                    return False, "The provided zip file is too large to process."
                zip_content.write(chunk)
            
            zip_content.seek(0)

            # Inspect zip file contents without extracting
            with zipfile.ZipFile(zip_content, 'r') as zf:
                total_uncompressed_size = 0
                num_files = 0
                file_list = []

                for file_info in zf.infolist():
                    num_files += 1
                    total_uncompressed_size += file_info.file_size
                    file_list.append(file_info.filename)

                    # Check for too many files or excessive uncompressed size
                    if num_files > self.max_files:
                        return False, "The provided zip could not be processed due to potential safety risk (too many files)."
                    if total_uncompressed_size > self.max_uncompressed_size:
                        return False, "The provided zip could not be processed due to potential safety risk (uncompressed size too large)."
                
                # Summarize file types within the zip
                file_counts = {}
                for f in file_list:
                    ext = f.split('.')[-1].lower() if '.' in f else 'other'
                    file_counts[ext] = file_counts.get(ext, 0) + 1
                
                count_str_parts = []
                for ext, count in file_counts.items():
                    count_str_parts.append(f"{count} .{ext}")

                return True, f"The provided zip file contains {num_files} file(s): {', '.join(count_str_parts)}."

        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading zip file: {e}")
            return False, "Could not download the zip file."
        except zipfile.BadZipFile:
            return False, "The provided file is not a valid zip file."
        except Exception as e:
            logging.error(f"An unexpected error occurred while handling zip file: {e}")
            return False, "An unexpected error occurred while processing the zip file."