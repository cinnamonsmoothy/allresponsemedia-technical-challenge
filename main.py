import argparse
import sys
from pathlib import Path
from src.kaggle_client import KaggleClient, KaggleAPIError
from src.data_processor import DataProcessor
from src.visualizer import Visualizer
from src.logger import get_logger, get_context_logger
from src.config import config

logger = get_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YouTube Trending Videos Analysis")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download even if data files already exist"
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory for visualizations"
    )
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Create context logger for this run
    ctx_logger = get_context_logger(__name__, {"run_id": f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"})

    ctx_logger.info("Starting YouTube Trending Videos Analysis")
    ctx_logger.info(f"Configuration: force_download={args.force_download}, skip_viz={args.skip_viz}")

    try:
        # Step 1: Initialize Kaggle client and download data
        ctx_logger.info("Step 1: Downloading data from Kaggle")
        client = KaggleClient()

        # Check for newer version
        latest_version = client.check_dataset_version()
        if latest_version and latest_version != config.kaggle.dataset_version:
            ctx_logger.warning(f"Newer dataset version available: {latest_version} (current: {config.kaggle.dataset_version})")

        # Download dataset
        csv_path, json_path = client.download_dataset(force_download=args.force_download)
        ctx_logger.info(f"Data downloaded - CSV: {csv_path}, JSON: {json_path}")

        # Step 2: Process and clean data
        ctx_logger.info("Step 2: Processing and cleaning data")
        processor = DataProcessor()
        processor.load_data(csv_path, json_path)
        processor.clean_data()

        # Get data summary
        summary = processor.get_data_summary()
        ctx_logger.info(f"Data processed - {summary['total_videos']} videos, {summary['total_categories']} categories")

        # Step 3: Generate insights
        ctx_logger.info("Step 3: Generating insights")

        # Get top videos by different metrics
        top_by_views = processor.get_top_videos('views', 10)
        top_by_engagement = processor.get_top_videos('engagement_rate', 10) if 'engagement_rate' in processor.videos_df.columns else None

        # Get category statistics
        category_stats = processor.get_category_stats()

        ctx_logger.info(f"Insights generated - Top video: '{top_by_views.iloc[0]['title'][:50]}...'")

        # Step 4: Create visualizations (unless skipped)
        if not args.skip_viz:
            ctx_logger.info("Step 4: Creating visualizations")

            visualizer = Visualizer()

            # Override output directory if specified
            if args.output_dir:
                visualizer.output_dir = Path(args.output_dir)
                visualizer.output_dir.mkdir(parents=True, exist_ok=True)

            # Create visualizations
            viz_paths = []

            # Category analysis
            if not category_stats.empty:
                path = visualizer.create_category_analysis(category_stats)
                if path:
                    viz_paths.append(path)

            # Top videos charts
            path = visualizer.create_top_videos_chart(top_by_views, 'views')
            if path:
                viz_paths.append(path)

            if top_by_engagement is not None:
                path = visualizer.create_top_videos_chart(top_by_engagement, 'engagement_rate')
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

            ctx_logger.info(f"Visualizations created - {len(viz_paths)} charts saved")
            for path in viz_paths:
                ctx_logger.info(f"{Path(path).name}")

        # Step 5: Print summary
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"Total Videos Analyzed: {summary['total_videos']:,}")
        print(f"Categories Found: {summary['total_categories']}")
        print(f"Total Views: {summary['total_views']/1e9:.1f}B")
        print(f"Average Views: {summary['avg_views']/1e6:.1f}M")

        if 'avg_engagement_rate' in summary:
            print(f"Average Engagement: {summary['avg_engagement_rate']:.2f}%")

        print(f"\nTop Video by Views:")
        print(f"   '{top_by_views.iloc[0]['title'][:60]}...'")
        print(f"   {top_by_views.iloc[0]['views']/1e6:.1f}M views")

        if not category_stats.empty:
            print(f"\nTop Category: {category_stats.index[0]}")
            print(f"   {category_stats.iloc[0]['video_count']} videos, {category_stats.iloc[0]['total_views']/1e9:.1f}B total views")

        if not args.skip_viz:
            print(f"\nVisualizations saved to: {visualizer.output_dir}")

        print("="*60)

        ctx_logger.info("Analysis completed successfully")
        return 0

    except KaggleAPIError as e:
        ctx_logger.error(f"Kaggle API error: {e}")
        print(f"\nError: {e}")
        print("Please check your Kaggle credentials and internet connection.")
        return 1

    except FileNotFoundError as e:
        ctx_logger.error(f"File not found: {e}")
        print(f"\nError: {e}")
        return 1

    except Exception as e:
        ctx_logger.error(f"Unexpected error: {e}")
        print(f"\nUnexpected error: {e}")
        print("Check the logs for more details.")
        return 1


if __name__ == "__main__":
    # Import pandas here to avoid circular imports
    import pandas as pd

    exit_code = main()
    sys.exit(exit_code)