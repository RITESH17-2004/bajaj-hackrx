import re
import unicodedata

def clean_escape_characters(text: str) -> str:
    """Remove ALL unwanted backslash escape characters comprehensively"""
    # Normalize Unicode characters first
    text = unicodedata.normalize('NFKD', text)
    
    # Handle specific escape sequences first
    text = text.replace('"', '"')     # " -> "
    text = text.replace("'", "'")     # ' -> '
    text = text.replace('\\\\', '\\') # double backslash -> single backslash
    text = text.replace('\n', ' ')    # newline -> space
    text = text.replace('\t', ' ')    # tab -> space
    text = text.replace('\r', '')     # carriage return -> remove
    text = text.replace('\f', ' ')    # form feed -> space
    text = text.replace('\b', ' ')    # backspace -> space
    text = text.replace('\v', ' ')    # vertical tab -> space

    
    # Remove any remaining backslash followed by non-whitespace character
    text = re.sub(r'\\([^\s\\])', r'\\\1', text)
    
    # Remove any standalone backslashes (including double backslashes converted above)
    text = re.sub(r'\\+', ' ', text)
    
    # Normalize quotes - convert curly quotes to straight quotes
    text = text.replace('“', '"').replace('”', '"')  # Smart quotes to regular
    text = text.replace('‘', "'").replace('’', "'")  # Smart single quotes to regular
    
    # Clean up multiple spaces created by replacements
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()
