"""
integration tests for the complete application workflow.
"""

import pytest
import json
import zipfile
import io
from unittest.mock import patch, Mock
from pathlib import Path

from src.kaggle_client import KaggleClient
from src.data_processor import DataProcessor
from src.visualizer import Visualizer


class TestEndToEndWorkflow:
    """test the complete application workflow."""
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_complete_workflow_simulation(self, mock_get, temp_dir):
        """test the complete workflow from download to visualization."""
        
        # Step 1: Set up mock data
        sample_csv_content = """video_id,title,category_id,views,likes,dislikes,comment_count
vid1,Test Video 1,10,1000000,50000,5000,1000
vid2,Test Video 2,24,500000,25000,2500,500
vid3,Test Video 3,10,2000000,100000,10000,2000
vid4,Test Video 4,1,750000,37500,3750,750
vid5,Test Video 5,24,1200000,60000,6000,1200"""
        
        sample_json_content = {
            "items": [
                {"id": "1", "snippet": {"title": "Film & Animation"}},
                {"id": "10", "snippet": {"title": "Music"}},
                {"id": "24", "snippet": {"title": "Entertainment"}}
            ]
        }
        
        # Create mock zip file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr("GBvideos.csv", sample_csv_content)
            zip_file.writestr("GB_category_id.json", json.dumps(sample_json_content))
        zip_content = zip_buffer.getvalue()
        
        # Step 2: Mock Kaggle API responses
        version_response = Mock()
        version_response.json.return_value = {"currentVersionNumber": 115}
        version_response.raise_for_status.return_value = None
        
        download_response = Mock()
        download_response.iter_content.return_value = [zip_content]
        download_response.raise_for_status.return_value = None
        
        info_response = Mock()
        info_response.json.return_value = {"title": "Test Dataset", "description": "Test"}
        info_response.raise_for_status.return_value = None
        
        mock_get.side_effect = [version_response, download_response, info_response]
        
        # Step 3: Set up credentials
        credentials = {"username": "test_user", "key": "test_key"}
        credentials_file = temp_dir / "kaggle.json"
        with open(credentials_file, 'w') as f:
            json.dump(credentials, f)
        
        # Step 4: Execute the complete workflow
        with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_creds_path, \
             patch('src.kaggle_client.config.paths.data_dir', temp_dir), \
             patch('src.visualizer.plt.savefig'), \
             patch('src.visualizer.plt.close'), \
             patch('src.visualizer.plt.subplots'), \
             patch('src.visualizer.plt.figure'):
            
            mock_creds_path.return_value = credentials_file
            
            # Initialize components
            client = KaggleClient()
            processor = DataProcessor()
            visualizer = Visualizer()
            visualizer.output_dir = temp_dir
            
            # Step 4a: Download data
            csv_path, json_path = client.download_dataset(force_download=True)
            
            assert csv_path.exists()
            assert json_path.exists()
            assert "Test Video 1" in csv_path.read_text()
            
            # Step 4b: Process data
            processor.load_data(csv_path, json_path)
            processor.clean_data()
            
            assert len(processor.videos_df) == 5
            assert 'category_name' in processor.videos_df.columns
            assert 'engagement_rate' in processor.videos_df.columns
            
            # Step 4c: Generate insights
            top_videos = processor.get_top_videos('views', 3)
            category_stats = processor.get_category_stats()
            summary = processor.get_data_summary()
            
            assert len(top_videos) == 3
            assert top_videos.iloc[0]['title'] == 'Test Video 3'  # Highest views
            assert len(category_stats) == 3
            assert summary['total_videos'] == 5
            
            # Step 4d: Create visualizations
            viz_paths = []
            
            path = visualizer.create_category_analysis(category_stats)
            if path:
                viz_paths.append(path)
            
            path = visualizer.create_top_videos_chart(top_videos, 'views')
            if path:
                viz_paths.append(path)
            
            path = visualizer.create_engagement_analysis(processor.videos_df)
            if path:
                viz_paths.append(path)
            
            path = visualizer.create_summary_dashboard(processor.videos_df, category_stats)
            if path:
                viz_paths.append(path)
            
            # Verify visualizations were created
            assert len(viz_paths) == 4
            
            # Step 5: Verify final results
            assert summary['total_views'] == 5450000  # Sum of all views
            assert summary['total_categories'] == 3
            assert category_stats.loc['Music']['video_count'] == 2  # 2 music videos
    
    def test_error_handling_workflow(self, temp_dir):
        """Test workflow with various error conditions."""
        
        # Test with missing credentials
        with pytest.raises(Exception):  # Should raise KaggleAPIError
            with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path:
                mock_path.side_effect = FileNotFoundError("No credentials")
                KaggleClient()
        
        # Test with invalid data
        processor = DataProcessor()
        
        # Create invalid CSV (missing required columns)
        invalid_csv = temp_dir / "invalid.csv"
        with open(invalid_csv, 'w') as f:
            f.write("video_id,title\nvid1,Test")  # Missing required columns
        
        # Create valid JSON
        valid_json = temp_dir / "valid.json"
        with open(valid_json, 'w') as f:
            json.dump({"items": [{"id": "1", "snippet": {"title": "Test"}}]}, f)
        
        processor.load_data(invalid_csv, valid_json)
        
        with pytest.raises(ValueError, match="Missing required columns"):
            processor._validate_data()
    
    def test_data_consistency_across_modules(self, sample_data_processor, temp_dir):
        """test that data remains consistent across different modules."""
        processor = sample_data_processor
        
        # Get data from processor
        original_data = processor.videos_df.copy()
        top_videos = processor.get_top_videos('views', 5)
        category_stats = processor.get_category_stats()
        
        # Verify data consistency
        assert len(top_videos) == len(original_data)  # All videos returned
        assert top_videos['views'].sum() == original_data['views'].sum()
        
        # Verify category stats consistency
        total_videos_in_stats = category_stats['video_count'].sum()
        assert total_videos_in_stats == len(original_data)
        
        total_views_in_stats = category_stats['total_views'].sum()
        assert total_views_in_stats == original_data['views'].sum()
    
    @patch('src.kaggle_client.requests.Session.get')
    def test_version_checking_workflow(self, mock_get, mock_kaggle_credentials):
        """test the version checking workflow."""
        
        # Mock different version scenarios
        scenarios = [
            {"currentVersionNumber": 115},  # Same version
            {"currentVersionNumber": 116},  # Newer version
            {"currentVersionNumber": 114},  # Older version
        ]
        
        with patch('src.kaggle_client.config.get_kaggle_credentials_path') as mock_path:
            mock_path.return_value = mock_kaggle_credentials
            
            client = KaggleClient()
            
            for scenario in scenarios:
                mock_response = Mock()
                mock_response.json.return_value = scenario
                mock_response.raise_for_status.return_value = None
                mock_get.return_value = mock_response
                
                version = client.check_dataset_version()
                assert version == str(scenario["currentVersionNumber"])
    
    def test_memory_usage_workflow(self, sample_data_processor):
        """test that the workflow doesn't consume excessive memory."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        processor = sample_data_processor
        
        # Perform memory-intensive operations
        for _ in range(10):
            top_videos = processor.get_top_videos('views', 5)
            category_stats = processor.get_category_stats()
            summary = processor.get_data_summary()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50 * 1024 * 1024
    
    def test_concurrent_operations(self, sample_data_processor):
        """test that multiple operations can be performed concurrently."""
        import threading
        import time
        
        processor = sample_data_processor
        results = {}
        errors = []
        
        def get_top_videos(metric, thread_id):
            try:
                result = processor.get_top_videos(metric, 3)
                results[f"{metric}_{thread_id}"] = len(result)
            except Exception as e:
                errors.append(e)
        
        def get_category_stats(thread_id):
            try:
                result = processor.get_category_stats()
                results[f"categories_{thread_id}"] = len(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t1 = threading.Thread(target=get_top_videos, args=('views', i))
            t2 = threading.Thread(target=get_category_stats, args=(i,))
            threads.extend([t1, t2])
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10  # 5 top_videos + 5 category_stats
        
        # Verify all operations returned expected results
        for key, value in results.items():
            if 'views' in key:
                assert value == 3  # Top 3 videos
            elif 'categories' in key:
                assert value == 3  # 3 categories in sample data


class TestPerformanceBenchmarks:
    """performance benchmarks for the application."""
    
    def test_data_processing_performance(self, sample_csv_file, sample_json_file):
        """test data processing performance."""
        import time
        
        processor = DataProcessor()
        
        start_time = time.time()
        
        # Load and process data
        processor.load_data(sample_csv_file, sample_json_file)
        processor.clean_data()
        
        # Generate insights
        processor.get_top_videos('views', 10)
        processor.get_category_stats()
        processor.get_data_summary()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete in reasonable time (< 1 second for small dataset)
        assert duration < 1.0
    
    def test_visualization_performance(self, sample_data_processor, temp_dir):
        """test visualization generation performance."""
        import time
        
        processor = sample_data_processor
        visualizer = Visualizer()
        visualizer.output_dir = temp_dir
        
        top_videos = processor.get_top_videos('views', 5)
        category_stats = processor.get_category_stats()
        
        with patch('src.visualizer.plt.savefig'), \
             patch('src.visualizer.plt.close'), \
             patch('src.visualizer.plt.subplots'), \
             patch('src.visualizer.plt.figure'):
            
            start_time = time.time()
            
            # Create all visualizations
            visualizer.create_category_analysis(category_stats)
            visualizer.create_top_videos_chart(top_videos, 'views')
            visualizer.create_engagement_analysis(processor.videos_df)
            visualizer.create_summary_dashboard(processor.videos_df, category_stats)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Should complete in reasonable time (< 2 seconds with mocked plotting)
            assert duration < 2.0
