import requests
import zipfile
from io import BytesIO
import logging
from typing import Tuple, Optional, List
from enum import Enum
import asyncio

class ValidationStatus(Enum):
    SAFE_TO_PROCESS = 1
    UNSAFE = 2
    ZIP_ARCHIVE = 3

class FileValidator:
    def __init__(self, max_zip_size=100 * 1024 * 1024, max_uncompressed_size=500 * 1024 * 1024, max_files=1000):
        self.max_zip_size = max_zip_size  # 100 MB
        self.max_uncompressed_size = max_uncompressed_size # 500 MB
        self.max_files = max_files

    async def validate_url(self, document_url: str) -> Tuple[ValidationStatus, Optional[str]]:
        if document_url.lower().endswith('.bin'):
            return ValidationStatus.UNSAFE, "The provided url points to a binary file not a readable document."

        if document_url.lower().endswith('.zip'):
            status, message = await self._handle_zip(document_url)
            if status: # is safe
                return ValidationStatus.ZIP_ARCHIVE, message
            else: # is unsafe
                return ValidationStatus.UNSAFE, message

        return ValidationStatus.SAFE_TO_PROCESS, None

    async def _handle_zip(self, zip_url: str) -> Tuple[bool, Optional[str]]:
        try:
            # Using a sync function in an executor to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._sync_handle_zip, zip_url)
        except Exception as e:
            logging.error(f"An unexpected error occurred while handling zip file: {e}")
            return False, "An unexpected error occurred while processing the zip file."

    def _sync_handle_zip(self, zip_url: str) -> Tuple[bool, Optional[str]]:
        try:
            response = requests.get(zip_url, stream=True)
            response.raise_for_status()

            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_zip_size:
                return False, "The provided zip file is too large to process."

            zip_content = BytesIO()
            size = 0
            for chunk in response.iter_content(1024*1024):
                size += len(chunk)
                if size > self.max_zip_size:
                    return False, "The provided zip file is too large to process."
                zip_content.write(chunk)
            
            zip_content.seek(0)

            with zipfile.ZipFile(zip_content, 'r') as zf:
                total_uncompressed_size = 0
                num_files = 0
                file_list = []

                for file_info in zf.infolist():
                    num_files += 1
                    total_uncompressed_size += file_info.file_size
                    file_list.append(file_info.filename)

                    if num_files > self.max_files:
                        return False, "The provided zip could not be processed due to potential safety risk (too many files)."
                    if total_uncompressed_size > self.max_uncompressed_size:
                        return False, "The provided zip could not be processed due to potential safety risk (uncompressed size too large)."
                
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
