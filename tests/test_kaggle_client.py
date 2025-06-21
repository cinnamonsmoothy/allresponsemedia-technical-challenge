"""
unit tests for the kaggle client module.
"""

import json
import pytest
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import zipfile
import io

from src.kaggle_client import KaggleClient, KaggleAPIError


class TestKaggleClient:
    """test cases for KaggleClient class."""
    
    def test_init_with_valid_credentials(self, mock_kaggle_credentials):
        """test client initialization with valid credentials."""
        with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path:
            mock_path.return_value = mock_kaggle_credentials
            
            client = KaggleClient()
            
            assert client.credentials["username"] == "test_user"
            assert client.credentials["key"] == "test_api_key_12345"
            assert client.session is not None
    
    def test_init_with_missing_credentials(self):
        """test client initialization with missing credentials file."""
        with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path:
            mock_path.side_effect = FileNotFoundError("Credentials not found")
            
            with pytest.raises(KaggleAPIError, match="Kaggle credentials not found"):
                KaggleClient()
    
    def test_init_with_invalid_json(self, temp_dir):
        """test client initialization with invalid JSON credentials."""
        invalid_file = temp_dir / "invalid.json"
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path:
            mock_path.return_value = invalid_file
            
            with pytest.raises(KaggleAPIError, match="Invalid JSON"):
                KaggleClient()
    
    def test_init_with_incomplete_credentials(self, temp_dir):
        """test client initialization with incomplete credentials."""
        incomplete_creds = {"username": "test_user"}  # missing key
        creds_file = temp_dir / "incomplete.json"
        
        with open(creds_file, 'w') as f:
            json.dump(incomplete_creds, f)
        
        with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path:
            mock_path.return_value = creds_file
            
            with pytest.raises(KaggleAPIError, match="Invalid kaggle.json format"):
                KaggleClient()
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_check_dataset_version_success(self, mock_get, mock_kaggle_client):
        """test successful dataset version checking."""
        mock_response = Mock()
        mock_response.json.return_value = {"currentVersionNumber": 116}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        version = mock_kaggle_client.check_dataset_version()
        
        assert version == "116"
        mock_get.assert_called_once()
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_check_dataset_version_failure(self, mock_get, mock_kaggle_client):
        """test dataset version checking with API failure."""
        mock_get.side_effect = Exception("API Error")
        
        version = mock_kaggle_client.check_dataset_version()
        
        assert version is None
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_get_dataset_info_success(self, mock_get, mock_kaggle_client):
        """test successful dataset info retrieval."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "title": "Test Dataset",
            "description": "Test Description"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        info = mock_kaggle_client.get_dataset_info()
        
        assert info["title"] == "Test Dataset"
        assert info["description"] == "Test Description"
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_get_dataset_info_failure(self, mock_get, mock_kaggle_client):
        """test dataset info retrieval with API failure."""
        mock_get.side_effect = Exception("API Error")
        
        info = mock_kaggle_client.get_dataset_info()
        
        assert info == {}
    
    def test_download_dataset_files_exist(self, mock_kaggle_client, temp_dir):
        """Test download when files already exist."""
        # Create existing files
        csv_path = temp_dir / "GBvideos.csv"
        json_path = temp_dir / "GB_category_id.json"
        csv_path.touch()
        json_path.touch()
        
        with patch('src.kaggle_client.config.paths.data_dir', temp_dir):
            result_csv, result_json = mock_kaggle_client.download_dataset()
            
            assert result_csv == csv_path
            assert result_json == json_path
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_download_dataset_success(self, mock_get, mock_kaggle_client, temp_dir):
        """test successful dataset download."""
        # Create a mock zip file content
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr("GBvideos.csv", "video_id,title\nvid1,Test Video")
            zip_file.writestr("GB_category_id.json", '{"1": "Music"}')
        zip_content = zip_buffer.getvalue()
        
        # Mock the response
        mock_response = Mock()
        mock_response.iter_content.return_value = [zip_content]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        with patch('src.kaggle_client.config.paths.data_dir', temp_dir):
            csv_path, json_path = mock_kaggle_client.download_dataset(force_download=True)
            
            assert csv_path.exists()
            assert json_path.exists()
            assert "Test Video" in csv_path.read_text()
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_download_dataset_network_error(self, mock_get, mock_kaggle_client):
        """test download with network error."""
        mock_get.side_effect = Exception("Network Error")
        
        with pytest.raises(KaggleAPIError, match="Unexpected error during download"):
            mock_kaggle_client.download_dataset(force_download=True)
    
    def test_extract_files_missing_csv(self, mock_kaggle_client, temp_dir):
        """test extraction when CSV file is missing from zip."""
        # Create zip without CSV file
        zip_path = temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr("GB_category_id.json", '{"1": "Music"}')
        
        csv_path = temp_dir / "GBvideos.csv"
        json_path = temp_dir / "GB_category_id.json"
        
        with pytest.raises(KaggleAPIError, match="CSV file.*not found"):
            mock_kaggle_client._extract_files(zip_path, csv_path, json_path)
    
    def test_extract_files_missing_json(self, mock_kaggle_client, temp_dir):
        """test extraction when JSON file is missing from zip."""
        # Create zip without JSON file
        zip_path = temp_dir / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            zip_file.writestr("GBvideos.csv", "video_id,title\nvid1,Test")
        
        csv_path = temp_dir / "GBvideos.csv"
        json_path = temp_dir / "GB_category_id.json"
        
        with pytest.raises(KaggleAPIError, match="JSON file.*not found"):
            mock_kaggle_client._extract_files(zip_path, csv_path, json_path)
    
    def test_extract_files_invalid_zip(self, mock_kaggle_client, temp_dir):
        """test extraction with invalid zip file."""
        # Create invalid zip file
        zip_path = temp_dir / "invalid.zip"
        with open(zip_path, 'w') as f:
            f.write("not a zip file")
        
        csv_path = temp_dir / "GBvideos.csv"
        json_path = temp_dir / "GB_category_id.json"
        
        with pytest.raises(KaggleAPIError, match="Invalid zip file"):
            mock_kaggle_client._extract_files(zip_path, csv_path, json_path)


class TestKaggleClientIntegration:
    """integration tests for KaggleClient."""
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_full_workflow_simulation(self, mock_get, mock_kaggle_credentials, temp_dir):
        """Test the complete workflow simulation."""
        # Mock version check response
        version_response = Mock()
        version_response.json.return_value = {"currentVersionNumber": 115}
        version_response.raise_for_status.return_value = None
        
        # Mock download response
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr("GBvideos.csv", "video_id,title,views\nvid1,Test Video,1000")
            zip_file.writestr("GB_category_id.json", '{"items": [{"id": "1", "snippet": {"title": "Music"}}]}')
        
        download_response = Mock()
        download_response.iter_content.return_value = [zip_buffer.getvalue()]
        download_response.raise_for_status.return_value = None
        
        # Configure mock to return different responses for different calls
        mock_get.side_effect = [version_response, download_response]
        
        with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path, \
             patch('src.kaggle_client.config.paths.data_dir', temp_dir):
            
            mock_path.return_value = mock_kaggle_credentials
            
            client = KaggleClient()
            
            # Check version
            version = client.check_dataset_version()
            assert version == "115"
            
            # Download dataset
            csv_path, json_path = client.download_dataset(force_download=True)
            
            assert csv_path.exists()
            assert json_path.exists()
            assert "Test Video" in csv_path.read_text()
            assert "Music" in json_path.read_text()
