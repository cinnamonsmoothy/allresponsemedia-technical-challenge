"""
Visualization module for YouTube trending video analysis.

Creates meaningful charts and graphs from processed data,
including category analysis, engagement metrics, and trends.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np

from .config import config
from .logger import get_logger

logger = get_logger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class Visualizer:
    """
    Creates visualizations for YouTube trending video data analysis.
    
    Generates various charts including category analysis, engagement metrics,
    top videos, and statistical distributions.
    """
    
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.figure_size = config.figure_size
        self.dpi = config.dpi
        self.output_dir = config.paths.output_dir
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"Visualizer initialized - Output dir: {self.output_dir}")
    
    def create_category_analysis(self, category_stats: pd.DataFrame, 
                               save_path: Optional[str] = None) -> str:
        """
        Create comprehensive category analysis visualizations.
        
        Args:
            category_stats: DataFrame with category statistics
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization file
        """
        if category_stats.empty:
            logger.warning("No category data provided for visualization")
            return ""
        
        logger.info("Creating category analysis visualization")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('YouTube Trending Videos - Category Analysis', fontsize=16, fontweight='bold')
        
        # 1. Video count by category (bar chart)
        top_categories = category_stats.head(10)
        axes[0, 0].bar(range(len(top_categories)), top_categories['video_count'])
        axes[0, 0].set_title('Number of Trending Videos by Category')
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Number of Videos')
        axes[0, 0].set_xticks(range(len(top_categories)))
        axes[0, 0].set_xticklabels(top_categories.index, rotation=45, ha='right')
        
        # Add value labels on bars
        for i, v in enumerate(top_categories['video_count']):
            axes[0, 0].text(i, v + 0.5, str(int(v)), ha='center', va='bottom')
        
        # 2. Total views by category (horizontal bar chart)
        axes[0, 1].barh(range(len(top_categories)), top_categories['total_views'])
        axes[0, 1].set_title('Total Views by Category')
        axes[0, 1].set_xlabel('Total Views (Millions)')
        axes[0, 1].set_ylabel('Category')
        axes[0, 1].set_yticks(range(len(top_categories)))
        axes[0, 1].set_yticklabels(top_categories.index)
        
        # Format x-axis to show millions
        axes[0, 1].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # 3. Average engagement rate by category (scatter plot)
        if 'avg_engagement_rate' in category_stats.columns:
            scatter = axes[1, 0].scatter(top_categories['avg_views'], 
                                       top_categories['avg_engagement_rate'],
                                       s=top_categories['video_count']*10,
                                       alpha=0.6)
            axes[1, 0].set_title('Engagement Rate vs Average Views')
            axes[1, 0].set_xlabel('Average Views')
            axes[1, 0].set_ylabel('Average Engagement Rate (%)')
            axes[1, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            
            # Add category labels
            for i, category in enumerate(top_categories.index):
                axes[1, 0].annotate(category, 
                                  (top_categories.iloc[i]['avg_views'], 
                                   top_categories.iloc[i]['avg_engagement_rate']),
                                  xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Category performance heatmap
        if len(top_categories) > 1:
            # Normalize data for heatmap
            heatmap_data = top_categories[['video_count', 'avg_views', 'avg_likes']].copy()
            heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
            
            sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=top_categories.index, ax=axes[1, 1])
            axes[1, 1].set_title('Category Performance Heatmap (Normalized)')
            axes[1, 1].set_xlabel('Category')
            axes[1, 1].set_ylabel('Metrics')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "category_analysis.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Category analysis saved to: {save_path}")
        return str(save_path)
    
    def create_top_videos_chart(self, top_videos: pd.DataFrame, 
                              metric: str = 'views',
                              save_path: Optional[str] = None) -> str:
        """
        Create a chart showing top videos by specified metric.
        
        Args:
            top_videos: DataFrame with top videos
            metric: Metric used for ranking
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization file
        """
        if top_videos.empty:
            logger.warning("No top videos data provided for visualization")
            return ""
        
        logger.info(f"Creating top videos chart for metric: {metric}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(top_videos))
        bars = ax.barh(y_pos, top_videos[metric])
        
        # Customize the chart
        ax.set_yticks(y_pos)
        ax.set_yticklabels([title[:50] + '...' if len(title) > 50 else title 
                           for title in top_videos['title']])
        ax.invert_yaxis()  # Top video at the top
        ax.set_xlabel(f'{metric.replace("_", " ").title()}')
        ax.set_title(f'Top {len(top_videos)} YouTube Videos by {metric.replace("_", " ").title()}')
        
        # Format x-axis based on metric
        if 'views' in metric.lower():
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        elif 'rate' in metric.lower():
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if 'views' in metric.lower():
                label = f'{width/1e6:.1f}M'
            elif 'rate' in metric.lower():
                label = f'{width:.1f}%'
            else:
                label = f'{width:,.0f}'
            
            ax.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                   label, ha='left', va='center')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / f"top_videos_{metric}.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Top videos chart saved to: {save_path}")
        return str(save_path)
    
    def create_engagement_analysis(self, videos_df: pd.DataFrame,
                                 save_path: Optional[str] = None) -> str:
        """
        Create engagement analysis visualizations.
        
        Args:
            videos_df: DataFrame with video data including engagement metrics
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization file
        """
        if videos_df.empty:
            logger.warning("No video data provided for engagement analysis")
            return ""
        
        logger.info("Creating engagement analysis visualization")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('YouTube Trending Videos - Engagement Analysis', fontsize=16, fontweight='bold')
        
        # 1. Views vs Likes scatter plot
        if 'likes' in videos_df.columns:
            axes[0, 0].scatter(videos_df['views'], videos_df['likes'], alpha=0.6)
            axes[0, 0].set_xlabel('Views')
            axes[0, 0].set_ylabel('Likes')
            axes[0, 0].set_title('Views vs Likes')
            axes[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
            axes[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e3:.0f}K'))
        
        # 2. Engagement rate distribution
        if 'engagement_rate' in videos_df.columns:
            axes[0, 1].hist(videos_df['engagement_rate'], bins=30, alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Engagement Rate (%)')
            axes[0, 1].set_ylabel('Number of Videos')
            axes[0, 1].set_title('Distribution of Engagement Rates')
            axes[0, 1].axvline(videos_df['engagement_rate'].mean(), color='red', 
                             linestyle='--', label=f'Mean: {videos_df["engagement_rate"].mean():.2f}%')
            axes[0, 1].legend()
        
        # 3. Like ratio distribution
        if 'like_ratio' in videos_df.columns:
            axes[1, 0].hist(videos_df['like_ratio'], bins=30, alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Like Ratio (%)')
            axes[1, 0].set_ylabel('Number of Videos')
            axes[1, 0].set_title('Distribution of Like Ratios')
            axes[1, 0].axvline(videos_df['like_ratio'].mean(), color='red', 
                             linestyle='--', label=f'Mean: {videos_df["like_ratio"].mean():.1f}%')
            axes[1, 0].legend()
        
        # 4. Views distribution (log scale)
        axes[1, 1].hist(np.log10(videos_df['views'] + 1), bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Log10(Views + 1)')
        axes[1, 1].set_ylabel('Number of Videos')
        axes[1, 1].set_title('Distribution of Views (Log Scale)')
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "engagement_analysis.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Engagement analysis saved to: {save_path}")
        return str(save_path)
    
    def create_summary_dashboard(self, videos_df: pd.DataFrame, 
                               category_stats: pd.DataFrame,
                               save_path: Optional[str] = None) -> str:
        """
        Create a comprehensive summary dashboard.
        
        Args:
            videos_df: DataFrame with video data
            category_stats: DataFrame with category statistics
            save_path: Optional custom save path
            
        Returns:
            Path to saved visualization file
        """
        logger.info("Creating summary dashboard")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle('YouTube Trending Videos - Summary Dashboard', fontsize=20, fontweight='bold')
        
        # Key metrics text
        ax_metrics = fig.add_subplot(gs[0, 0])
        ax_metrics.axis('off')
        
        total_videos = len(videos_df)
        total_views = videos_df['views'].sum()
        avg_engagement = videos_df['engagement_rate'].mean() if 'engagement_rate' in videos_df.columns else 0
        top_category = category_stats.index[0] if not category_stats.empty else "Unknown"
        
        metrics_text = f"""
        KEY METRICS
        
        Total Videos: {total_videos:,}
        Total Views: {total_views/1e9:.1f}B
        Avg Engagement: {avg_engagement:.2f}%
        Top Category: {top_category}
        """
        
        ax_metrics.text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Top categories pie chart
        if not category_stats.empty:
            ax_pie = fig.add_subplot(gs[0, 1:])
            top_5_categories = category_stats.head(5)
            ax_pie.pie(top_5_categories['video_count'], labels=top_5_categories.index, autopct='%1.1f%%')
            ax_pie.set_title('Top 5 Categories by Video Count')
        
        # Views distribution
        ax_views = fig.add_subplot(gs[1, :])
        ax_views.hist(videos_df['views'], bins=50, alpha=0.7, edgecolor='black')
        ax_views.set_xlabel('Views')
        ax_views.set_ylabel('Number of Videos')
        ax_views.set_title('Distribution of Video Views')
        ax_views.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        
        # Category performance
        if not category_stats.empty:
            ax_cat = fig.add_subplot(gs[2, :])
            top_10_categories = category_stats.head(10)
            x_pos = np.arange(len(top_10_categories))
            
            ax_cat.bar(x_pos, top_10_categories['total_views'], alpha=0.7)
            ax_cat.set_xlabel('Category')
            ax_cat.set_ylabel('Total Views')
            ax_cat.set_title('Total Views by Category (Top 10)')
            ax_cat.set_xticks(x_pos)
            ax_cat.set_xticklabels(top_10_categories.index, rotation=45, ha='right')
            ax_cat.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e9:.1f}B'))
        
        # Save the plot
        if save_path is None:
            save_path = self.output_dir / "summary_dashboard.png"
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Summary dashboard saved to: {save_path}")
        return str(save_path)
