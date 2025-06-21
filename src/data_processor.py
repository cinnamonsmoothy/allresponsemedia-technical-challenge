"""
Data processing module for YouTube trending video analysis.

Handles CSV/JSON parsing, data cleaning, transformation,
and preparation for visualization.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .config import config
from .logger import get_logger

logger = get_logger(__name__)


class DataProcessor:
    """
    Processes YouTube trending video data for analysis and visualization.
    
    Handles CSV parsing, JSON category mapping, data cleaning,
    and transformation for meaningful insights.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.videos_df: Optional[pd.DataFrame] = None
        self.categories: Optional[Dict[str, str]] = None
        logger.info("Data processor initialized")
    
    def load_data(self, csv_path: Path, json_path: Path) -> None:
        """
        load and parse the CSV and JSON data files.
        """
        logger.info(f"Loading data from CSV: {csv_path}, JSON: {json_path}")
        
        try:
            # Load videos CSV
            self.videos_df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.videos_df)} video records")
            
            # Load categories JSON
            with open(json_path, 'r', encoding='utf-8') as f:
                categories_data = json.load(f)
            
            # Extract category mapping from JSON structure
            self.categories = self._parse_categories(categories_data)
            logger.info(f"Loaded {len(self.categories)} video categories")
            
            # Validate data
            self._validate_data()
            
        except FileNotFoundError as e:
            logger.error(f"Data file not found: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {e}")
            raise ValueError(f"Empty CSV file: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading data: {e}")
            raise
    
    def _parse_categories(self, categories_data: Dict) -> Dict[str, str]:
        """
        parse the categories JSON structure.
        """
        categories = {}
        
        # Handle different possible JSON structures
        if "items" in categories_data:
            # Standard YouTube API format
            for item in categories_data["items"]:
                cat_id = item["id"]
                cat_name = item["snippet"]["title"]
                categories[cat_id] = cat_name
        else:
            # Simple key-value format
            categories = categories_data
        
        logger.debug(f"Parsed categories: {list(categories.keys())}")
        return categories
    
    def _validate_data(self) -> None:
        """
        validate the loaded data for required columns and basic integrity.
        """
        if self.videos_df is None:
            raise ValueError("Videos data not loaded")
        
        # Check for required columns
        required_columns = ['video_id', 'title', 'category_id', 'views', 'likes', 'dislikes']
        missing_columns = [col for col in required_columns if col not in self.videos_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check for empty dataframe
        if len(self.videos_df) == 0:
            raise ValueError("No video data found")
        
        logger.info("Data validation passed")
    
    def clean_data(self) -> None:
        """
        clean and preprocess the video data.
        
        handles missing values, data type conversions, and outlier removal.
        """
        if self.videos_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        logger.info("Starting data cleaning")
        original_count = len(self.videos_df)
        
        # Remove duplicates
        self.videos_df = self.videos_df.drop_duplicates(subset=['video_id'])
        logger.info(f"Removed {original_count - len(self.videos_df)} duplicate videos")
        
        # Convert numeric columns
        numeric_columns = ['views', 'likes', 'dislikes', 'comment_count']
        for col in numeric_columns:
            if col in self.videos_df.columns:
                self.videos_df[col] = pd.to_numeric(self.videos_df[col], errors='coerce')
        
        # Handle missing values
        self.videos_df = self.videos_df.dropna(subset=['video_id', 'title', 'category_id'])
        
        # Fill missing numeric values with 0
        for col in numeric_columns:
            if col in self.videos_df.columns:
                self.videos_df[col] = self.videos_df[col].fillna(0)
        
        # Convert category_id to string for mapping
        self.videos_df['category_id'] = self.videos_df['category_id'].astype(str)
        
        # Add category names
        if self.categories:
            self.videos_df['category_name'] = self.videos_df['category_id'].map(self.categories)
            self.videos_df['category_name'] = self.videos_df['category_name'].fillna('Unknown')
        
        # Calculate engagement metrics
        self._calculate_engagement_metrics()
        
        logger.info(f"Data cleaning completed. Final dataset: {len(self.videos_df)} videos")
    
    def _calculate_engagement_metrics(self) -> None:
        """Calculate additional engagement metrics for analysis."""
        if 'likes' in self.videos_df.columns and 'dislikes' in self.videos_df.columns:
            # Engagement rate (likes + dislikes) / views
            self.videos_df['total_engagement'] = self.videos_df['likes'] + self.videos_df['dislikes']
            self.videos_df['engagement_rate'] = (
                self.videos_df['total_engagement'] / self.videos_df['views'].replace(0, 1)
            ) * 100
            
            # Like ratio
            self.videos_df['like_ratio'] = (
                self.videos_df['likes'] / self.videos_df['total_engagement'].replace(0, 1)
            ) * 100
        
        logger.debug("Engagement metrics calculated")
    
    def get_top_videos(self, metric: str = 'views', n: int = 10) -> pd.DataFrame:
        """
        Get top N videos by specified metric.
        
        Args:
            metric: Metric to sort by ('views', 'likes', 'engagement_rate', etc.)
            n: Number of top videos to return
            
        Returns:
            DataFrame with top N videos
        """
        if self.videos_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if metric not in self.videos_df.columns:
            raise ValueError(f"Metric '{metric}' not found in data")
        
        top_videos = self.videos_df.nlargest(n, metric)
        logger.info(f"Retrieved top {n} videos by {metric}")
        
        return top_videos
    
    def get_category_stats(self) -> pd.DataFrame:
        """
        Get statistics by video category.
        
        Returns:
            DataFrame with category-wise statistics
        """
        if self.videos_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'category_name' not in self.videos_df.columns:
            logger.warning("Category names not available")
            return pd.DataFrame()
        
        category_stats = self.videos_df.groupby('category_name').agg({
            'video_id': 'count',
            'views': ['mean', 'sum'],
            'likes': ['mean', 'sum'],
            'engagement_rate': 'mean'
        }).round(2)
        
        # Flatten column names
        category_stats.columns = ['_'.join(col).strip() for col in category_stats.columns]
        category_stats = category_stats.rename(columns={
            'video_id_count': 'video_count',
            'views_mean': 'avg_views',
            'views_sum': 'total_views',
            'likes_mean': 'avg_likes',
            'likes_sum': 'total_likes',
            'engagement_rate_mean': 'avg_engagement_rate'
        })
        
        category_stats = category_stats.sort_values('total_views', ascending=False)
        logger.info(f"Generated statistics for {len(category_stats)} categories")
        
        return category_stats
    
    def get_data_summary(self) -> Dict:
        """
        Get a summary of the loaded data.
        
        Returns:
            Dictionary with data summary statistics
        """
        if self.videos_df is None:
            return {"error": "No data loaded"}
        
        summary = {
            "total_videos": len(self.videos_df),
            "total_categories": self.videos_df['category_name'].nunique() if 'category_name' in self.videos_df.columns else 0,
            "total_views": self.videos_df['views'].sum(),
            "total_likes": self.videos_df['likes'].sum() if 'likes' in self.videos_df.columns else 0,
            "avg_views": self.videos_df['views'].mean(),
            "avg_engagement_rate": self.videos_df['engagement_rate'].mean() if 'engagement_rate' in self.videos_df.columns else 0,
            "date_range": {
                "columns": list(self.videos_df.columns),
                "shape": self.videos_df.shape
            }
        }
        
        logger.info("Generated data summary")
        return summary
