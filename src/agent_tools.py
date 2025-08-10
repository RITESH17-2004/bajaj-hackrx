import logging
import re
from typing import Dict, Any
import requests

def api_call(instruction: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs an API call based on the provided instruction.
    Expects a 'url' in the instruction dictionary.
    Sanitizes the URL and handles potential request errors.
    """
    logging.info(f"Making API call with instruction: {instruction}")

    url = instruction.get("url")
    if not url:
        raise ValueError("URL not found in instruction.")

    # Sanitize the URL to ensure it's valid and points to the correct domain
    url = url.replace("register hackrx in", "register.hackrx.in")
    url = url.replace(" ", "") # Remove any spaces
    
    # Ensure the URL starts with the correct base path
    if not url.startswith("https://register.hackrx.in"):
        if url.startswith("/"):
            url = "https://register.hackrx.in" + url
        else:
            url = "https://register.hackrx.in/" + url 

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return {"api_response": response.json()}
    except requests.exceptions.RequestException as e:
        logging.error(f"API call failed: {e}")
        raise # Re-raise the exception after logging

def text_parser(instruction: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts information from the document text using a regex pattern.
    Expects a 'pattern' in the instruction dictionary and 'document_text' in current_state.
    """
    logging.info(f"Parsing text with instruction: {instruction}")

    text = current_state.get("document_text")
    if not text:
        raise ValueError("Document text not found in current state.")

    pattern = instruction.get("pattern")
    if not pattern:
        raise ValueError("Pattern not found in instruction.")

    try:
        # Search for the pattern in the text, ignoring case and allowing dot to match newlines
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError(f"Pattern not found in text: {pattern}")

        return {"parsed_text": match.groups()}
    except Exception as e:
        logging.error(f"Error during text parsing: {e}")
        raise # Re-raise the exception after logging

def conditional_logic(instruction: Dict[str, Any], current_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes different actions based on a given condition.
    Expects 'condition', 'true_action', and 'false_action' in the instruction dictionary.
    Note: The condition evaluation here is a simplified example and would need a more robust implementation.
    """
    logging.info(f"Executing conditional logic with instruction: {instruction}")

    condition = instruction.get("condition")
    true_action = instruction.get("true_action")
    false_action = instruction.get("false_action")

    if not all([condition, true_action, false_action]):
        raise ValueError("Missing required parameters for conditional_logic.")

    # This is a simplified example. A real implementation would need a more
    # robust and secure way to evaluate the condition.
    if eval(condition, {}, current_state): # Evaluate the condition against the current state
        return {"next_action": true_action}
    else:
        return {"next_action": false_action}
