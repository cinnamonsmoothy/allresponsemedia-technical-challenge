"""
unit tests for the visualizer module.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, Mock

from src.visualizer import Visualizer


class TestVisualizer:
    """test cases for Visualizer class."""
    
    def test_init(self, temp_dir):
        """test visualizer initialization."""
        visualizer = Visualizer()
        
        assert visualizer.figure_size == (12, 8)
        assert visualizer.dpi == 300
        assert visualizer.output_dir.exists()
    
    def test_init_custom_output_dir(self, temp_dir):
        """test visualizer initialization with custom output directory."""
        visualizer = Visualizer()
        visualizer.output_dir = temp_dir
        
        assert visualizer.output_dir == temp_dir
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    @patch('src.visualizer.plt.subplots')
    def test_create_category_analysis_success(self, mock_subplots, mock_close, mock_savefig, sample_visualizer):
        """Test successful category analysis creation."""
        # Create sample category stats
        category_stats = pd.DataFrame({
            'video_count': [10, 8, 5],
            'total_views': [1000000, 800000, 500000],
            'avg_views': [100000, 100000, 100000],
            'avg_likes': [5000, 4000, 2500],
            'avg_engagement_rate': [5.0, 4.5, 3.0]
        }, index=['Music', 'Entertainment', 'Gaming'])
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the method
        result_path = sample_visualizer.create_category_analysis(category_stats)
        
        # Verify calls
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify result
        assert result_path.endswith('category_analysis.png')
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    def test_create_category_analysis_empty_data(self, mock_close, mock_savefig, sample_visualizer):
        """test category analysis with empty data."""
        empty_stats = pd.DataFrame()
        
        result_path = sample_visualizer.create_category_analysis(empty_stats)
        
        assert result_path == ""
        mock_savefig.assert_not_called()
        mock_close.assert_not_called()
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    @patch('src.visualizer.plt.subplots')
    def test_create_top_videos_chart_success(self, mock_subplots, mock_close, mock_savefig, sample_visualizer):
        """Test successful top videos chart creation."""
        # Create sample top videos data
        top_videos = pd.DataFrame({
            'title': ['Video 1', 'Video 2', 'Video 3'],
            'views': [1000000, 800000, 600000],
            'engagement_rate': [5.0, 4.5, 4.0]
        })
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_ax = Mock()
        mock_bars = [Mock(), Mock(), Mock()]
        mock_ax.barh.return_value = mock_bars
        mock_subplots.return_value = (mock_fig, mock_ax)
        
        # Configure bar mocks
        for i, bar in enumerate(mock_bars):
            bar.get_width.return_value = top_videos.iloc[i]['views']
            bar.get_y.return_value = i
            bar.get_height.return_value = 0.8
        
        # Test the method
        result_path = sample_visualizer.create_top_videos_chart(top_videos, 'views')
        
        # Verify calls
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify result
        assert result_path.endswith('top_videos_views.png')
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    def test_create_top_videos_chart_empty_data(self, mock_close, mock_savefig, sample_visualizer):
        """test top videos chart with empty data."""
        empty_videos = pd.DataFrame()
        
        result_path = sample_visualizer.create_top_videos_chart(empty_videos, 'views')
        
        assert result_path == ""
        mock_savefig.assert_not_called()
        mock_close.assert_not_called()
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    @patch('src.visualizer.plt.subplots')
    def test_create_engagement_analysis_success(self, mock_subplots, mock_close, mock_savefig, sample_visualizer):
        """Test successful engagement analysis creation."""
        # Create sample video data
        videos_df = pd.DataFrame({
            'views': [1000000, 800000, 600000],
            'likes': [50000, 40000, 30000],
            'engagement_rate': [5.0, 4.5, 4.0],
            'like_ratio': [90.0, 88.0, 85.0]
        })
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        # Test the method
        result_path = sample_visualizer.create_engagement_analysis(videos_df)
        
        # Verify calls
        mock_subplots.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify result
        assert result_path.endswith('engagement_analysis.png')
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    def test_create_engagement_analysis_empty_data(self, mock_close, mock_savefig, sample_visualizer):
        """test engagement analysis with empty data."""
        empty_videos = pd.DataFrame()
        
        result_path = sample_visualizer.create_engagement_analysis(empty_videos)
        
        assert result_path == ""
        mock_savefig.assert_not_called()
        mock_close.assert_not_called()
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    @patch('src.visualizer.plt.figure')
    def test_create_summary_dashboard_success(self, mock_figure, mock_close, mock_savefig, sample_visualizer):
        """test successful summary dashboard creation."""
        # Create sample data
        videos_df = pd.DataFrame({
            'views': [1000000, 800000, 600000],
            'engagement_rate': [5.0, 4.5, 4.0],
            'category_name': ['Music', 'Entertainment', 'Gaming']
        })
        
        category_stats = pd.DataFrame({
            'video_count': [10, 8, 5],
            'total_views': [1000000, 800000, 500000]
        }, index=['Music', 'Entertainment', 'Gaming'])
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_gs = Mock()
        mock_fig.add_gridspec.return_value = mock_gs
        mock_figure.return_value = mock_fig
        
        # Mock subplot creation
        mock_subplot = Mock()
        mock_fig.add_subplot.return_value = mock_subplot
        
        # Test the method
        result_path = sample_visualizer.create_summary_dashboard(videos_df, category_stats)
        
        # Verify calls
        mock_figure.assert_called_once()
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()
        
        # Verify result
        assert result_path.endswith('summary_dashboard.png')
    
    def test_create_summary_dashboard_custom_path(self, sample_visualizer, temp_dir):
        """test summary dashboard with custom save path."""
        videos_df = pd.DataFrame({'views': [1000000], 'engagement_rate': [5.0], 'category_name': ['Music']})
        category_stats = pd.DataFrame({'video_count': [1], 'total_views': [1000000]}, index=['Music'])
        
        custom_path = temp_dir / "custom_dashboard.png"
        
        with patch('src.visualizer.plt.savefig') as mock_savefig, \
             patch('src.visualizer.plt.close'), \
             patch('src.visualizer.plt.figure'):
            
            result_path = sample_visualizer.create_summary_dashboard(
                videos_df, category_stats, str(custom_path)
            )
            
            assert result_path == str(custom_path)
            mock_savefig.assert_called_once()


class TestVisualizerIntegration:
    """integration tests for Visualizer."""
    
    @patch('src.visualizer.plt.savefig')
    @patch('src.visualizer.plt.close')
    @patch('src.visualizer.plt.subplots')
    @patch('src.visualizer.plt.figure')
    @patch('src.visualizer.sns.heatmap')
    def test_full_visualization_workflow(self, mock_heatmap, mock_figure, mock_subplots, 
                                       mock_close, mock_savefig, sample_data_processor, temp_dir):
        """Test the complete visualization workflow."""
        # Use the sample data processor
        processor = sample_data_processor
        
        # Create visualizer
        visualizer = Visualizer()
        visualizer.output_dir = temp_dir
        
        # Mock matplotlib objects
        mock_fig = Mock()
        mock_axes = [[Mock(), Mock()], [Mock(), Mock()]]
        mock_subplots.return_value = (mock_fig, mock_axes)
        mock_figure.return_value = mock_fig
        
        mock_gs = Mock()
        mock_fig.add_gridspec.return_value = mock_gs
        mock_subplot = Mock()
        mock_fig.add_subplot.return_value = mock_subplot
        
        # Mock bar chart returns
        mock_bars = [Mock() for _ in range(5)]
        for i, bar in enumerate(mock_bars):
            bar.get_width.return_value = 1000000 - i * 100000
            bar.get_y.return_value = i
            bar.get_height.return_value = 0.8
        mock_axes[0][0].barh.return_value = mock_bars
        
        # Get data for visualizations
        top_videos = processor.get_top_videos('views', 5)
        category_stats = processor.get_category_stats()
        
        # Create all visualizations
        viz_paths = []
        
        # Category analysis
        path = visualizer.create_category_analysis(category_stats)
        if path:
            viz_paths.append(path)
        
        # Top videos chart
        path = visualizer.create_top_videos_chart(top_videos, 'views')
        if path:
            viz_paths.append(path)
        
        # Engagement analysis
        path = visualizer.create_engagement_analysis(processor.videos_df)
        if path:
            viz_paths.append(path)
        
        # Summary dashboard
        path = visualizer.create_summary_dashboard(processor.videos_df, category_stats)
        if path:
            viz_paths.append(path)
        
        # Verify that visualizations were created
        assert len(viz_paths) == 4
        
        # Verify matplotlib was called appropriately
        assert mock_savefig.call_count == 4
        assert mock_close.call_count == 4
    
    def test_visualization_file_naming(self, sample_visualizer):
        """test that visualization files are named correctly."""
        # Create minimal test data
        videos_df = pd.DataFrame({'views': [1000], 'engagement_rate': [5.0], 'category_name': ['Music']})
        category_stats = pd.DataFrame({'video_count': [1], 'total_views': [1000]}, index=['Music'])
        top_videos = pd.DataFrame({'title': ['Test'], 'views': [1000]})
        
        with patch('src.visualizer.plt.savefig') as mock_savefig, \
             patch('src.visualizer.plt.close'), \
             patch('src.visualizer.plt.subplots'), \
             patch('src.visualizer.plt.figure'):
            
            # Test different chart types
            path1 = sample_visualizer.create_category_analysis(category_stats)
            path2 = sample_visualizer.create_top_videos_chart(top_videos, 'views')
            path3 = sample_visualizer.create_top_videos_chart(top_videos, 'engagement_rate')
            path4 = sample_visualizer.create_engagement_analysis(videos_df)
            path5 = sample_visualizer.create_summary_dashboard(videos_df, category_stats)
            
            # Verify file names
            assert path1.endswith('category_analysis.png')
            assert path2.endswith('top_videos_views.png')
            assert path3.endswith('top_videos_engagement_rate.png')
            assert path4.endswith('engagement_analysis.png')
            assert path5.endswith('summary_dashboard.png')
