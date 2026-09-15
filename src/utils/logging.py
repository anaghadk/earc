import logging
import re
import sys
import os
from typing import Optional

class APIKeyFilter(logging.Filter):
    """Filter to redact API keys from logs."""
    
    API_KEY_REGEX = re.compile(r'(api[_-]?key[\'"]?\s*[:=]\s*[\'"]?)[a-zA-Z0-9\-_]{15,}([\'"]?)', re.IGNORECASE)
    BEARER_REGEX = re.compile(r'(Bearer\s+)[a-zA-Z0-9\-_]{15,}', re.IGNORECASE)

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.API_KEY_REGEX.sub(r'\1***REDACTED***\2', record.msg)
            record.msg = self.BEARER_REGEX.sub(r'\1***REDACTED***', record.msg)
        return True

def setup_logging(level: str, log_file: Optional[str] = None, run_id: Optional[str] = None) -> logging.Logger:
    """Set up structured logging with API key filtering."""
    logger = logging.getLogger()
    
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
        
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    log_format = '%(asctime)s - %(name)s - %(levelname)s'
    if run_id:
        log_format += f' - [Run: {run_id}]'
    log_format += ' - %(message)s'
    
    formatter = logging.Formatter(log_format)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(APIKeyFilter())
    logger.addHandler(console_handler)
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(APIKeyFilter())
        logger.addHandler(file_handler)
        
    return logger

def get_logger(name: str) -> logging.Logger:
    """Factory to get a module-level logger."""
    return logging.getLogger(name)
