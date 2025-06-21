"""
Structured logging module for Kaggle API Ingestion Service.

Provides JSON-structured logging for production monitoring and
human-readable logging for development.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

from .config import config


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON-formatted log string
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "lineno", "funcName", "created",
                "msecs", "relativeCreated", "thread", "threadName",
                "processName", "process", "getMessage", "exc_info",
                "exc_text", "stack_info"
            }:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ContextLogger:
    """
    Logger wrapper that adds context to all log messages.
    
    Useful for adding request IDs, user IDs, or other contextual information
    to all logs within a specific scope.
    """
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Add context to log message."""
        # Create a new record with context
        extra = kwargs.pop('extra', {})
        extra.update(self.context)

        self.logger.log(level, msg, *args, extra=extra, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)


def setup_logging() -> logging.Logger:
    """
    Set up structured logging based on configuration.
    
    Returns:
        Configured logger instance
    """
    # Create root logger
    logger = logging.getLogger("kaggle_ingestion")
    logger.setLevel(getattr(logging, config.logging.level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Set formatter based on configuration
    if config.logging.format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if specified
    if config.logging.log_file:
        log_file_path = Path(config.logging.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"kaggle_ingestion.{name}")


def get_context_logger(name: str, context: Dict[str, Any]) -> ContextLogger:
    """
    Get a context logger that adds context to all log messages.
    
    Args:
        name: Logger name
        context: Context dictionary to add to all log messages
        
    Returns:
        ContextLogger instance
    """
    logger = get_logger(name)
    return ContextLogger(logger, context)


# Initialize logging when module is imported
setup_logging()
