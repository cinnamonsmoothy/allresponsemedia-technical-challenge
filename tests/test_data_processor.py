"""
unit tests for the data processor module.
"""

import json
import pytest
import pandas as pd
from pathlib import Path

from src.data_processor import DataProcessor


class TestDataProcessor:
    """test cases for DataProcessor class."""
    
    def test_init(self):
        """test processor initialization."""
        processor = DataProcessor()
        
        assert processor.videos_df is None
        assert processor.categories is None
    
    def test_load_data_success(self, sample_csv_file, sample_json_file):
        """test successful data loading."""
        processor = DataProcessor()
        processor.load_data(sample_csv_file, sample_json_file)
        
        assert processor.videos_df is not None
        assert len(processor.videos_df) == 5
        assert processor.categories is not None
        assert len(processor.categories) == 3
        assert processor.categories["10"] == "Music"
    
    def test_load_data_missing_csv(self, sample_json_file, temp_dir):
        """test loading with missing CSV file."""
        processor = DataProcessor()
        missing_csv = temp_dir / "missing.csv"
        
        with pytest.raises(FileNotFoundError):
            processor.load_data(missing_csv, sample_json_file)
    
    def test_load_data_missing_json(self, sample_csv_file, temp_dir):
        """test loading with missing JSON file."""
        processor = DataProcessor()
        missing_json = temp_dir / "missing.json"
        
        with pytest.raises(FileNotFoundError):
            processor.load_data(sample_csv_file, missing_json)
    
    def test_load_data_empty_csv(self, sample_json_file, temp_dir):
        """test loading with empty CSV file."""
        processor = DataProcessor()
        empty_csv = temp_dir / "empty.csv"
        empty_csv.touch()  # Create empty file
        
        with pytest.raises(ValueError, match="Empty CSV file"):
            processor.load_data(empty_csv, sample_json_file)
    
    def test_load_data_invalid_json(self, sample_csv_file, temp_dir):
        """test loading with invalid JSON file."""
        processor = DataProcessor()
        invalid_json = temp_dir / "invalid.json"
        
        with open(invalid_json, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(ValueError, match="Invalid JSON format"):
            processor.load_data(sample_csv_file, invalid_json)
    
    def test_parse_categories_standard_format(self):
        """test parsing categories in standard YouTube API format."""
        processor = DataProcessor()
        categories_data = {
            "items": [
                {"id": "1", "snippet": {"title": "Film & Animation"}},
                {"id": "10", "snippet": {"title": "Music"}}
            ]
        }
        
        categories = processor._parse_categories(categories_data)
        
        assert categories["1"] == "Film & Animation"
        assert categories["10"] == "Music"
        assert len(categories) == 2
    
    def test_parse_categories_simple_format(self):
        """test parsing categories in simple key-value format."""
        processor = DataProcessor()
        categories_data = {
            "1": "Film & Animation",
            "10": "Music"
        }
        
        categories = processor._parse_categories(categories_data)
        
        assert categories["1"] == "Film & Animation"
        assert categories["10"] == "Music"
        assert len(categories) == 2
    
    def test_validate_data_success(self, sample_csv_file, sample_json_file):
        """test successful data validation."""
        processor = DataProcessor()
        processor.load_data(sample_csv_file, sample_json_file)
        
        # Should not raise any exception
        processor._validate_data()
    
    def test_validate_data_missing_columns(self, temp_dir, sample_json_file):
        """test validation with missing required columns."""
        # Create CSV with missing columns
        incomplete_data = pd.DataFrame({
            'video_id': ['vid1', 'vid2'],
            'title': ['Video 1', 'Video 2']
            # Missing required columns: category_id, views, likes, dislikes
        })
        
        csv_path = temp_dir / "incomplete.csv"
        incomplete_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        processor.load_data(csv_path, sample_json_file)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            processor._validate_data()
    
    def test_validate_data_empty_dataframe(self, temp_dir, sample_json_file):
        """test validation with empty dataframe."""
        # Create empty CSV with headers
        empty_data = pd.DataFrame(columns=['video_id', 'title', 'category_id', 'views', 'likes', 'dislikes'])
        
        csv_path = temp_dir / "empty_with_headers.csv"
        empty_data.to_csv(csv_path, index=False)
        
        processor = DataProcessor()
        processor.load_data(csv_path, sample_json_file)
        
        with pytest.raises(ValueError, match="No video data found"):
            processor._validate_data()
    
    def test_clean_data_success(self, sample_data_processor):
        """Test successful data cleaning."""
        processor = sample_data_processor
        
        # Check that data was cleaned
        assert len(processor.videos_df) == 5  # No duplicates in sample data
        assert 'category_name' in processor.videos_df.columns
        assert 'engagement_rate' in processor.videos_df.columns
        assert 'like_ratio' in processor.videos_df.columns
        
        # Check category mapping
        music_videos = processor.videos_df[processor.videos_df['category_name'] == 'Music']
        assert len(music_videos) == 2
    
    def test_clean_data_with_duplicates(self, sample_csv_file, sample_json_file):
        """test data cleaning with duplicate videos."""
        processor = DataProcessor()
        processor.load_data(sample_csv_file, sample_json_file)
        
        # Add duplicate row
        duplicate_row = processor.videos_df.iloc[0].copy()
        processor.videos_df = pd.concat([processor.videos_df, duplicate_row.to_frame().T], ignore_index=True)
        
        original_count = len(processor.videos_df)
        processor.clean_data()
        
        # Should have removed the duplicate
        assert len(processor.videos_df) < original_count
    
    def test_clean_data_no_data_loaded(self):
        """test cleaning when no data is loaded."""
        processor = DataProcessor()
        
        with pytest.raises(ValueError, match="No data loaded"):
            processor.clean_data()
    
    def test_calculate_engagement_metrics(self, sample_data_processor):
        """test engagement metrics calculation."""
        processor = sample_data_processor
        
        # Check that engagement metrics were calculated
        assert 'total_engagement' in processor.videos_df.columns
        assert 'engagement_rate' in processor.videos_df.columns
        assert 'like_ratio' in processor.videos_df.columns
        
        # Verify calculations for first row
        first_row = processor.videos_df.iloc[0]
        expected_total_engagement = first_row['likes'] + first_row['dislikes']
        expected_engagement_rate = (expected_total_engagement / first_row['views']) * 100
        expected_like_ratio = (first_row['likes'] / expected_total_engagement) * 100
        
        assert first_row['total_engagement'] == expected_total_engagement
        assert abs(first_row['engagement_rate'] - expected_engagement_rate) < 0.01
        assert abs(first_row['like_ratio'] - expected_like_ratio) < 0.01
    
    def test_get_top_videos_by_views(self, sample_data_processor):
        """test getting top videos by views."""
        processor = sample_data_processor
        
        top_videos = processor.get_top_videos('views', 3)
        
        assert len(top_videos) == 3
        assert top_videos.iloc[0]['title'] == 'Test Video 3'  # Highest views
        assert top_videos.iloc[0]['views'] == 2000000
        
        # Check that results are sorted in descending order
        views = top_videos['views'].tolist()
        assert views == sorted(views, reverse=True)
    
    def test_get_top_videos_by_engagement(self, sample_data_processor):
        """test getting top videos by engagement rate."""
        processor = sample_data_processor
        
        top_videos = processor.get_top_videos('engagement_rate', 2)
        
        assert len(top_videos) == 2
        # Check that results are sorted in descending order
        engagement_rates = top_videos['engagement_rate'].tolist()
        assert engagement_rates == sorted(engagement_rates, reverse=True)
    
    def test_get_top_videos_invalid_metric(self, sample_data_processor):
        """test getting top videos with invalid metric."""
        processor = sample_data_processor
        
        with pytest.raises(ValueError, match="Metric 'invalid_metric' not found"):
            processor.get_top_videos('invalid_metric', 5)
    
    def test_get_top_videos_no_data(self):
        """Test getting top videos when no data is loaded."""
        processor = DataProcessor()
        
        with pytest.raises(ValueError, match="No data loaded"):
            processor.get_top_videos('views', 5)
    
    def test_get_category_stats(self, sample_data_processor):
        """test getting category statistics."""
        processor = sample_data_processor
        
        category_stats = processor.get_category_stats()
        
        assert len(category_stats) == 3  # 3 categories in sample data
        assert 'video_count' in category_stats.columns
        assert 'avg_views' in category_stats.columns
        assert 'total_views' in category_stats.columns
        
        # Check Music category (should have 2 videos)
        music_stats = category_stats.loc['Music']
        assert music_stats['video_count'] == 2
        assert music_stats['total_views'] == 3000000  # 1M + 2M
    
    def test_get_category_stats_no_categories(self, sample_csv_file, sample_json_file):
        """Test getting category stats when categories are not available."""
        processor = DataProcessor()
        processor.load_data(sample_csv_file, sample_json_file)
        
        # Remove category_name column
        if 'category_name' in processor.videos_df.columns:
            processor.videos_df = processor.videos_df.drop('category_name', axis=1)
        
        category_stats = processor.get_category_stats()
        
        assert category_stats.empty
    
    def test_get_category_stats_no_data(self):
        """test getting category stats when no data is loaded."""
        processor = DataProcessor()
        
        with pytest.raises(ValueError, match="No data loaded"):
            processor.get_category_stats()
    
    def test_get_data_summary(self, sample_data_processor):
        """test getting data summary."""
        processor = sample_data_processor
        
        summary = processor.get_data_summary()
        
        assert summary['total_videos'] == 5
        assert summary['total_categories'] == 3
        assert summary['total_views'] == 5450000  # Sum of all views
        assert summary['avg_views'] == 1090000  # Average views
        assert 'avg_engagement_rate' in summary
        assert 'date_range' in summary
        assert summary['date_range']['shape'] == (5, 10)  # 5 rows, 10 columns after processing
    
    def test_get_data_summary_no_data(self):
        """test getting data summary when no data is loaded."""
        processor = DataProcessor()
        
        summary = processor.get_data_summary()
        
        assert summary == {"error": "No data loaded"}


class TestDataProcessorIntegration:
    """integration tests for DataProcessor."""
    
    def test_full_processing_workflow(self, sample_csv_file, sample_json_file):
        """test the complete data processing workflow."""
        processor = DataProcessor()
        
        # Load data
        processor.load_data(sample_csv_file, sample_json_file)
        assert processor.videos_df is not None
        assert processor.categories is not None
        
        # Clean data
        processor.clean_data()
        assert 'category_name' in processor.videos_df.columns
        assert 'engagement_rate' in processor.videos_df.columns
        
        # Get insights
        top_videos = processor.get_top_videos('views', 3)
        category_stats = processor.get_category_stats()
        summary = processor.get_data_summary()
        
        assert len(top_videos) == 3
        assert len(category_stats) == 3
        assert summary['total_videos'] == 5
        
        # Verify data consistency
        assert summary['total_views'] == processor.videos_df['views'].sum()
        assert summary['total_categories'] == processor.videos_df['category_name'].nunique()
