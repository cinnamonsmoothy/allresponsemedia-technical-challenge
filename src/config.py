"""
configuration module for kaggl api ingestion service.
// what this files handles // 

- environment variables 
- API settings
- project constants.
"""

import os
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class KaggleConfig:
    """configuration for kaggle api interactions."""
    
    base_url: str = "https://www.kaggle.com/api/v1"
    owner_slug: str = "datasnaek"
    dataset_slug: str = "youtube-new"
    dataset_version: str = "115"
    csv_file_name: str = "GBvideos.csv"
    json_file_name: str = "GB_category_id.json"
    
    @property
    def download_url(self) -> str:
        """Generate the complete download URL for the dataset."""
        return f"{self.base_url}/datasets/download/{self.owner_slug}/{self.dataset_slug}?datasetVersionNumber={self.dataset_version}"


@dataclass
class PathConfig:
    """configuration for file paths and directories."""
    
    # Project root directory
    project_root: Path = Path(__file__).parent.parent
    
    # Data directories
    data_dir: Path = project_root / "data"
    output_dir: Path = project_root / "outputs"
    
    # Kaggle credentials
    kaggle_config_dir: Path = Path.home() / ".kaggle"
    kaggle_config_file: Path = kaggle_config_dir / "kaggle.json"
    
    # Local kaggle.json fallback (for development)
    local_kaggle_file: Path = project_root / "kaggle.json"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)


@dataclass
class LoggingConfig:
    """Configuration for structured logging."""
    
    level: str = os.getenv("LOG_LEVEL", "INFO")
    format_type: str = os.getenv("LOG_FORMAT", "json")  # json or text
    log_file: Optional[str] = os.getenv("LOG_FILE", None)


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Sub-configurations
    kaggle: KaggleConfig = KaggleConfig()
    paths: PathConfig = PathConfig()
    logging: LoggingConfig = LoggingConfig()
    
    # Application settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Data processing settings
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    timeout_seconds: int = int(os.getenv("TIMEOUT_SECONDS", "30"))
    
    # Visualization settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    
    def get_kaggle_credentials_path(self) -> Path:
        """
        get path to kaggle credentials file.
        """
        if self.paths.kaggle_config_file.exists():
            return self.paths.kaggle_config_file
        elif self.paths.local_kaggle_file.exists():
            return self.paths.local_kaggle_file
        else:
            raise FileNotFoundError(
                f"Kaggle credentials not found. Please place kaggle.json in either:\n"
                f"  - {self.paths.kaggle_config_file}\n"
                f"  - {self.paths.local_kaggle_file}"
            )


# Global configuration instance
config = AppConfig()
