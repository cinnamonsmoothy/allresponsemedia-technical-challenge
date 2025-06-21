"""
unit tests for the logging module.
"""

import json
import logging
import pytest
from unittest.mock import patch, Mock
from io import StringIO

from src.logger import JSONFormatter, ContextLogger, setup_logging, get_logger, get_context_logger


class TestJSONFormatter:
    """test cases for JSONFormatter class."""
    
    def test_format_basic_record(self):
        """test formatting a basic log record."""
        formatter = JSONFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        # Format the record
        formatted = formatter.format(record)
        
        # Parse the JSON
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line"] == 42
        assert "timestamp" in log_data
    
    def test_format_with_exception(self):
        """test formatting a log record with exception info."""
        formatter = JSONFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys
            exc_info = sys.exc_info()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="/test/path.py",
            lineno=42,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Error occurred"
        assert "exception" in log_data
        assert "ValueError: Test exception" in log_data["exception"]
    
    def test_format_with_extra_fields(self):
        """test formatting a log record with extra fields."""
        formatter = JSONFormatter()
        
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        
        # Add extra fields
        record.user_id = "12345"
        record.request_id = "req_abc123"
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["user_id"] == "12345"
        assert log_data["request_id"] == "req_abc123"


class TestContextLogger:
    """test cases for ContextLogger class."""
    
    def test_init(self):
        """test context logger initialization."""
        base_logger = logging.getLogger("test")
        context = {"user_id": "123", "session_id": "abc"}
        
        ctx_logger = ContextLogger(base_logger, context)
        
        assert ctx_logger.logger == base_logger
        assert ctx_logger.context == context
    
    def test_log_with_context(self):
        """test logging with context information."""
        # Create a string buffer to capture log output
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(JSONFormatter())
        
        base_logger = logging.getLogger("test_context")
        base_logger.setLevel(logging.INFO)
        base_logger.handlers.clear()
        base_logger.addHandler(handler)
        
        context = {"user_id": "123", "operation": "test"}
        ctx_logger = ContextLogger(base_logger, context)
        
        # Log a message
        ctx_logger.info("Test message")
        
        # Get the logged output
        log_output = log_stream.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["message"] == "Test message"
        assert log_data["user_id"] == "123"
        assert log_data["operation"] == "test"
    
    def test_different_log_levels(self):
        """test context logger with different log levels."""
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(JSONFormatter())
        
        base_logger = logging.getLogger("test_levels")
        base_logger.setLevel(logging.DEBUG)
        base_logger.handlers.clear()
        base_logger.addHandler(handler)
        
        context = {"component": "test"}
        ctx_logger = ContextLogger(base_logger, context)
        
        # Test different log levels
        ctx_logger.debug("Debug message")
        ctx_logger.info("Info message")
        ctx_logger.warning("Warning message")
        ctx_logger.error("Error message")
        ctx_logger.critical("Critical message")
        
        # Check that all messages were logged
        log_output = log_stream.getvalue()
        log_lines = [line for line in log_output.strip().split('\n') if line]
        
        assert len(log_lines) == 5
        
        # Check log levels
        levels = [json.loads(line)["level"] for line in log_lines]
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestLoggingSetup:
    """test cases for logging setup functions."""
    
    @patch('src.logger.config')
    def test_setup_logging_json_format(self, mock_config):
        """test logging setup with JSON format."""
        mock_config.logging.level = "INFO"
        mock_config.logging.format_type = "json"
        mock_config.logging.log_file = None
        
        logger = setup_logging()
        
        assert logger.name == "kaggle_ingestion"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0].formatter, JSONFormatter)
    
    @patch('src.logger.config')
    def test_setup_logging_text_format(self, mock_config):
        """test logging setup with text format."""
        mock_config.logging.level = "DEBUG"
        mock_config.logging.format_type = "text"
        mock_config.logging.log_file = None
        
        logger = setup_logging()
        
        assert logger.level == logging.DEBUG
        assert not isinstance(logger.handlers[0].formatter, JSONFormatter)
    
    @patch('src.logger.config')
    def test_setup_logging_with_file(self, mock_config, temp_dir):
        """test logging setup with file output."""
        log_file = temp_dir / "test.log"
        
        mock_config.logging.level = "INFO"
        mock_config.logging.format_type = "json"
        mock_config.logging.log_file = str(log_file)
        
        logger = setup_logging()
        
        # Should have console handler + file handler
        assert len(logger.handlers) == 2
    
    def test_get_logger(self):
        """test getting a named logger."""
        logger = get_logger("test_module")
        
        assert logger.name == "kaggle_ingestion.test_module"
    
    def test_get_context_logger(self):
        """test getting a context logger."""
        context = {"module": "test", "version": "1.0"}
        ctx_logger = get_context_logger("test_module", context)
        
        assert isinstance(ctx_logger, ContextLogger)
        assert ctx_logger.context == context
        assert ctx_logger.logger.name == "kaggle_ingestion.test_module"


class TestLoggingIntegration:
    """integration tests for logging functionality."""
    
    def test_full_logging_workflow(self, temp_dir):
        """test the complete logging workflow."""
        # Set up logging with file output
        log_file = temp_dir / "integration_test.log"
        
        with patch('src.logger.config') as mock_config:
            mock_config.logging.level = "INFO"
            mock_config.logging.format_type = "json"
            mock_config.logging.log_file = str(log_file)
            
            # Set up logging
            setup_logging()
            
            # Get loggers
            regular_logger = get_logger("integration_test")
            ctx_logger = get_context_logger("integration_test", {"test_id": "12345"})
            
            # Log some messages
            regular_logger.info("Regular log message")
            ctx_logger.info("Context log message")
            
            # Check that file was created and contains logs
            assert log_file.exists()
            
            log_content = log_file.read_text()
            log_lines = [line for line in log_content.strip().split('\n') if line]
            
            assert len(log_lines) >= 2
            
            # Parse and verify log entries
            regular_log = json.loads(log_lines[0])
            context_log = json.loads(log_lines[1])
            
            assert regular_log["message"] == "Regular log message"
            assert context_log["message"] == "Context log message"
            assert context_log["test_id"] == "12345"
    
    def test_logger_hierarchy(self):
        """test that logger hierarchy works correctly."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")
        
        assert child_logger.name == "kaggle_ingestion.parent.child"
        assert child_logger.parent.name == "kaggle_ingestion.parent"
    
    def test_logging_performance(self):
        """test that logging doesn't significantly impact performance."""
        import time
        
        logger = get_logger("performance_test")
        
        # Time logging operations
        start_time = time.time()
        
        for i in range(1000):
            logger.info(f"Performance test message {i}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 1000 log operations in reasonable time (< 1 second)
        assert duration < 1.0
