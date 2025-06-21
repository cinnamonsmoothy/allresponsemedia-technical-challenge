"""
kaggle api client for downloading datasets

// what this file handles // 
- authentication
- dataset 
- downloads 
- version checking,
- implements retry logic with proper error handling.
"""

import json
import time
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .config import config
from .logger import get_logger

logger = get_logger(__name__)


class KaggleAPIError(Exception):
    """custom exception for kaggle api errors."""
    pass


class KaggleClient:
    """
    client for interacting with the kaggle api.
    
    handles authentication, dataset downloads, and version management.
    """
    
    def __init__(self):
        """initialize the kaggle client with authentication."""
        self.credentials = self._load_credentials()
        self.session = self._create_session()
        logger.info(f"Kaggle client initialized for user: {self.credentials['username']}")
    
    def _load_credentials(self) -> Dict[str, str]:
        """
        load kaggle api credentials from kaggle.json file.
        """
        try:
            credentials_path = config.get_kaggle_credentials_path()
            logger.debug(f"Loading credentials from: {credentials_path}")
            
            with open(credentials_path, 'r') as f:
                credentials = json.load(f)
            
            if "username" not in credentials or "key" not in credentials:
                raise KaggleAPIError("invalid kaggle.json format. must contain 'username' and 'key'")
            
            logger.info("credentials loaded successfully")
            return credentials
            
        except FileNotFoundError as e:
            logger.error(f"kaggle credentials not found: {e}")
            raise KaggleAPIError(f"kaggle credentials not found: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"invalid JSON in kaggle.json: {e}")
            raise KaggleAPIError(f"invalid JSON in kaggle.json: {e}")
    
    def _create_session(self) -> requests.Session:
        """
        create a requests session with retry strategy and authentication.
        """
        session = requests.Session()
        
        # Set up authentication
        session.auth = (self.credentials["username"], self.credentials["key"])
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set timeout
        session.timeout = config.timeout_seconds
        
        return session
    
    def check_dataset_version(self) -> Optional[str]:
        """
        check the latest version of the dataset.
        """
        try:
            url = f"{config.kaggle.base_url}/datasets/view/{config.kaggle.owner_slug}/{config.kaggle.dataset_slug}"
            logger.debug(f"Checking dataset version at: {url}")
            
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            latest_version = str(data.get("currentVersionNumber", config.kaggle.dataset_version))
            
            logger.info(f"Dataset version checked - Current: {config.kaggle.dataset_version}, Latest: {latest_version}")
            
            return latest_version
            
        except Exception as e:
            logger.warning(f"Could not check dataset version: {e}")
            return None
    
    def download_dataset(self, force_download: bool = False) -> Tuple[Path, Path]:
        """
        download the dataset files if they don't exist or if forced.
        """
        csv_path = config.paths.data_dir / config.kaggle.csv_file_name
        json_path = config.paths.data_dir / config.kaggle.json_file_name
        
        # Check if files already exist
        if not force_download and csv_path.exists() and json_path.exists():
            logger.info(f"Dataset files already exist, skipping download - CSV: {csv_path}, JSON: {json_path}")
            return csv_path, json_path

        logger.info(f"Starting dataset download - {config.kaggle.owner_slug}/{config.kaggle.dataset_slug} v{config.kaggle.dataset_version}")
        
        try:
            # Download the dataset
            response = self.session.get(config.kaggle.download_url, stream=True)
            response.raise_for_status()
            
            # Save to temporary zip file
            zip_path = config.paths.data_dir / "temp_dataset.zip"
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Dataset downloaded to: {zip_path}")

            # Extract the specific files we need
            self._extract_files(zip_path, csv_path, json_path)

            # Clean up zip file
            zip_path.unlink()

            logger.info(f"Dataset extraction completed - CSV: {csv_path}, JSON: {json_path}")
            
            return csv_path, json_path
            
        except requests.RequestException as e:
            logger.error(f"Failed to download dataset: {e}")
            raise KaggleAPIError(f"Failed to download dataset: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            raise KaggleAPIError(f"Unexpected error during download: {e}")
    
    def _extract_files(self, zip_path: Path, csv_path: Path, json_path: Path) -> None:
        """
        extract specific files from the downloaded zip.
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # List all files in the zip
                file_list = zip_ref.namelist()
                logger.debug(f"Files in zip: {file_list}")
                
                # Extract CSV file
                csv_found = False
                for file_name in file_list:
                    if file_name.endswith(config.kaggle.csv_file_name):
                        with zip_ref.open(file_name) as source, open(csv_path, 'wb') as target:
                            target.write(source.read())
                        csv_found = True
                        break
                
                # Extract JSON file
                json_found = False
                for file_name in file_list:
                    if file_name.endswith(config.kaggle.json_file_name):
                        with zip_ref.open(file_name) as source, open(json_path, 'wb') as target:
                            target.write(source.read())
                        json_found = True
                        break
                
                if not csv_found:
                    raise KaggleAPIError(f"CSV file {config.kaggle.csv_file_name} not found in dataset")
                if not json_found:
                    raise KaggleAPIError(f"JSON file {config.kaggle.json_file_name} not found in dataset")
                
        except zipfile.BadZipFile as e:
            logger.error(f"Invalid zip file: {e}")
            raise KaggleAPIError(f"Invalid zip file: {e}")
    
    def get_dataset_info(self) -> Dict:
        """
        get information about the dataset
        """
        try:
            url = f"{config.kaggle.base_url}/datasets/view/{config.kaggle.owner_slug}/{config.kaggle.dataset_slug}"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"Dataset info retrieved: {data.get('title', 'Unknown')}")

            return data

        except Exception as e:
            logger.warning(f"Could not retrieve dataset info: {e}")
            return {}
