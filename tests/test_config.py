"""
unit tests for configuration module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch
from src.config import KaggleConfig, PathConfig, LoggingConfig, AppConfig


class TestKaggleConfig:
    """test cases for KaggleConfig class."""
    
    def test_default_values(self):
        """tst default configuration values."""
        config = KaggleConfig()
        
        assert config.base_url == "https://www.kaggle.com/api/v1"
        assert config.owner_slug == "datasnaek"
        assert config.dataset_slug == "youtube-new"
        assert config.dataset_version == "115"
        assert config.csv_file_name == "GBvideos.csv"
        assert config.json_file_name == "GB_category_id.json"
    
    def test_download_url_generation(self):
        """test download URL generation."""
        config = KaggleConfig()
        expected_url = "https://www.kaggle.com/api/v1/datasets/download/datasnaek/youtube-new?datasetVersionNumber=115"
        
        assert config.download_url == expected_url
    
    def test_custom_values(self):
        """Test configuration with custom values."""
        config = KaggleConfig(
            owner_slug="custom_owner",
            dataset_slug="custom_dataset",
            dataset_version="999"
        )
        
        assert config.owner_slug == "custom_owner"
        assert config.dataset_slug == "custom_dataset"
        assert config.dataset_version == "999"
        
        expected_url = "https://www.kaggle.com/api/v1/datasets/download/custom_owner/custom_dataset?datasetVersionNumber=999"
        assert config.download_url == expected_url


class TestPathConfig:
    """test cases for PathConfig class."""
    
    def test_default_paths(self):
        """Test default path configuration."""
        config = PathConfig()
        
        assert config.project_root.name == "kaggle_api_ingestion_simple"
        assert config.data_dir.name == "data"
        assert config.output_dir.name == "outputs"
        assert config.kaggle_config_dir == Path.home() / ".kaggle"
        assert config.kaggle_config_file.name == "kaggle.json"
        assert config.local_kaggle_file.name == "kaggle.json"
    
    def test_directory_creation(self, temp_dir):
        """test that directories are created during initialization."""
        # Create a custom PathConfig with temp directory
        config = PathConfig()
        config.data_dir = temp_dir / "test_data"
        config.output_dir = temp_dir / "test_outputs"
        
        # Trigger post_init by creating new instance
        config.__post_init__()
        
        assert config.data_dir.exists()
        assert config.output_dir.exists()


class TestLoggingConfig:
    """test cases for LoggingConfig class."""
    
    def test_default_values(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        
        assert config.level == "INFO"
        assert config.format_type == "json"
        assert config.log_file is None
    
    @patch.dict(os.environ, {
        'LOG_LEVEL': 'DEBUG',
        'LOG_FORMAT': 'text',
        'LOG_FILE': '/tmp/test.log'
    })
    def test_environment_variables(self):
        """Test configuration from environment variables."""
        config = LoggingConfig()
        
        assert config.level == "DEBUG"
        assert config.format_type == "text"
        assert config.log_file == "/tmp/test.log"


class TestAppConfig:
    """test cases for AppConfig class."""
    
    def test_default_values(self):
        """Test default application configuration."""
        config = AppConfig()
        
        assert isinstance(config.kaggle, KaggleConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert config.environment == "development"
        assert config.debug is False
        assert config.max_retries == 3
        assert config.timeout_seconds == 30
        assert config.figure_size == (12, 8)
        assert config.dpi == 300
    
    @patch.dict(os.environ, {
        'ENVIRONMENT': 'production',
        'DEBUG': 'true',
        'MAX_RETRIES': '5',
        'TIMEOUT_SECONDS': '60'
    })
    def test_environment_variables(self):
        """Test configuration from environment variables."""
        config = AppConfig()
        
        assert config.environment == "production"
        assert config.debug is True
        assert config.max_retries == 5
        assert config.timeout_seconds == 60
    
    def test_get_kaggle_credentials_path_standard(self, temp_dir):
        """Test getting Kaggle credentials from standard location."""
        config = AppConfig()
        
        # Create standard credentials file
        standard_path = temp_dir / ".kaggle" / "kaggle.json"
        standard_path.parent.mkdir(parents=True)
        standard_path.touch()
        
        # Mock the standard path
        config.paths.kaggle_config_file = standard_path
        config.paths.local_kaggle_file = temp_dir / "kaggle.json"
        
        result = config.get_kaggle_credentials_path()
        assert result == standard_path
    
    def test_get_kaggle_credentials_path_local(self, temp_dir):
        """test getting Kaggle credentials from local location."""
        config = AppConfig()
        
        # Create local credentials file
        local_path = temp_dir / "kaggle.json"
        local_path.touch()
        
        # Mock the paths
        config.paths.kaggle_config_file = temp_dir / ".kaggle" / "kaggle.json"  # Doesn't exist
        config.paths.local_kaggle_file = local_path
        
        result = config.get_kaggle_credentials_path()
        assert result == local_path
    
    def test_get_kaggle_credentials_path_not_found(self, temp_dir):
        """test getting Kaggle credentials when file doesn't exist."""
        config = AppConfig()
        
        # Mock paths to non-existent files
        config.paths.kaggle_config_file = temp_dir / ".kaggle" / "kaggle.json"
        config.paths.local_kaggle_file = temp_dir / "kaggle.json"
        
        with pytest.raises(FileNotFoundError, match="Kaggle credentials not found"):
            config.get_kaggle_credentials_path()


class TestConfigIntegration:
    """integration tests for configuration module."""
    
    def test_global_config_instance(self):
        """test that global config instance is properly initialized."""
        from src.config import config
        
        assert isinstance(config, AppConfig)
        assert isinstance(config.kaggle, KaggleConfig)
        assert isinstance(config.paths, PathConfig)
        assert isinstance(config.logging, LoggingConfig)
    
    def test_config_consistency(self):
        """test that configuration values are consistent across modules."""
        from src.config import config
        
        # Test that paths are consistent
        assert config.paths.project_root.exists()
        assert config.kaggle.download_url.startswith("https://")
        
        # Test that numeric values are reasonable
        assert config.max_retries > 0
        assert config.timeout_seconds > 0
        assert config.dpi > 0
        assert len(config.figure_size) == 2
    
    @patch.dict(os.environ, {
        'LOG_LEVEL': 'WARNING',
        'ENVIRONMENT': 'test',
        'MAX_RETRIES': '1'
    })
    def test_environment_override(self):
        """test that environment variables properly override defaults."""
        # Import config after setting environment variables
        from src.config import AppConfig
        
        config = AppConfig()
        
        assert config.logging.level == "WARNING"
        assert config.environment == "test"
        assert config.max_retries == 1
