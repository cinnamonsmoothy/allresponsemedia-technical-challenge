"""
pytest test suite
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil
from src.config import config
from src.kaggle_client import KaggleClient
from src.data_processor import DataProcessor
from src.visualizer import Visualizer


@pytest.fixture
def temp_dir():
    """create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_kaggle_credentials(temp_dir):
    """Create mock Kaggle credentials file."""
    credentials = {
        "username": "test_user",
        "key": "test_api_key_12345"
    }
    
    credentials_file = temp_dir / "kaggle.json"
    with open(credentials_file, 'w') as f:
        json.dump(credentials, f)
    
    return credentials_file


@pytest.fixture
def sample_csv_data():
    """create sample CSV data for testing."""
    data = {
        'video_id': ['vid1', 'vid2', 'vid3', 'vid4', 'vid5'],
        'title': [
            'Test Video 1',
            'Test Video 2',
            'Test Video 3',
            'Test Video 4',
            'Test Video 5'
        ],
        'category_id': ['10', '24', '10', '1', '24'],
        'views': [1000000, 500000, 2000000, 750000, 1200000],
        'likes': [50000, 25000, 100000, 37500, 60000],
        'dislikes': [5000, 2500, 10000, 3750, 6000],
        'comment_count': [1000, 500, 2000, 750, 1200]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_json_data():
    """create sample JSON category data for testing."""
    return {
        "items": [
            {
                "id": "1",
                "snippet": {"title": "Film & Animation"}
            },
            {
                "id": "10", 
                "snippet": {"title": "Music"}
            },
            {
                "id": "24",
                "snippet": {"title": "Entertainment"}
            }
        ]
    }


@pytest.fixture
def sample_csv_file(temp_dir, sample_csv_data):
    """create a sample CSV file for testing."""
    csv_path = temp_dir / "test_videos.csv"
    sample_csv_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_json_file(temp_dir, sample_json_data):
    """create a sample JSON file for testing."""
    json_path = temp_dir / "test_categories.json"
    with open(json_path, 'w') as f:
        json.dump(sample_json_data, f)
    return json_path


@pytest.fixture
def mock_requests_response():
    """create a mock requests response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {
        "currentVersionNumber": 115,
        "title": "Test Dataset"
    }
    return mock_response


@pytest.fixture
def mock_kaggle_client(mock_kaggle_credentials):
    """create a mock Kaggle client for testing."""
    with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path:
        mock_path.return_value = mock_kaggle_credentials
        client = KaggleClient()
        return client


@pytest.fixture
def sample_data_processor(sample_csv_file, sample_json_file):
    """create a data processor with sample data loaded."""
    processor = DataProcessor()
    processor.load_data(sample_csv_file, sample_json_file)
    processor.clean_data()
    return processor


@pytest.fixture
def sample_visualizer(temp_dir):
    """create a visualizer with temporary output directory."""
    visualizer = Visualizer()
    visualizer.output_dir = temp_dir
    return visualizer


@pytest.fixture
def mock_matplotlib():
    """mock matplotlib to prevent actual plot generation during tests."""
    with patch('matplotlib.pyplot.savefig') as mock_savefig, \
         patch('matplotlib.pyplot.close') as mock_close, \
         patch('matplotlib.pyplot.subplots') as mock_subplots:
        
        # Create mock figure and axes
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        yield {
            'savefig': mock_savefig,
            'close': mock_close,
            'subplots': mock_subplots,
            'fig': mock_fig,
            'ax': mock_ax
        }


# Test data constants
TEST_VIDEO_DATA = {
    'total_videos': 5,
    'categories': ['Music', 'Entertainment', 'Film & Animation'],
    'top_video_title': 'Test Video 3',
    'top_video_views': 2000000
}

TEST_CATEGORY_STATS = {
    'Music': {'video_count': 2, 'total_views': 3000000},
    'Entertainment': {'video_count': 2, 'total_views': 1700000},
    'Film & Animation': {'video_count': 1, 'total_views': 750000}
}
