#!/usr/bin/env python3
"""
Automatic Dataset Downloader Tool
Downloads datasets from various sources to D drive automatically
"""

import os
import sys
import requests
import zipfile
import tarfile
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import logging
from datetime import datetime
import hashlib

class DatasetDownloader:
    def __init__(self, base_dir="D:/datasets"):
        """
        Initialize the dataset downloader
        
        Args:
            base_dir (str): Base directory to store datasets (default: D:/datasets)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Common dataset sources
        self.dataset_sources = {
            "kaggle": "https://www.kaggle.com/datasets/",
            "uci": "https://archive.ics.uci.edu/ml/datasets/",
            "opendata": "https://data.gov/",
            "github": "https://github.com/",
            "huggingface": "https://huggingface.co/datasets/"
        }
        
        # Create necessary directories
        self.create_directory_structure()
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"downloader_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_directory_structure(self):
        """Create the directory structure for organizing datasets"""
        directories = [
            "raw",
            "processed", 
            "temp",
            "metadata",
            "logs"
        ]
        
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
            
    def download_file(self, url, destination, chunk_size=8192):
        """
        Download a file from URL with progress bar
        
        Args:
            url (str): URL to download from
            destination (Path): Destination file path
            chunk_size (int): Chunk size for download
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
                            
            self.logger.info(f"Successfully downloaded: {destination}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {str(e)}")
            return False
            
    def extract_archive(self, archive_path, extract_to):
        """
        Extract compressed archives (zip, tar.gz, etc.)
        
        Args:
            archive_path (Path): Path to the archive file
            extract_to (Path): Directory to extract to
        """
        try:
            extract_to.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                self.logger.warning(f"Unknown archive format: {archive_path.suffix}")
                return False
                
            self.logger.info(f"Successfully extracted: {archive_path} to {extract_to}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting {archive_path}: {str(e)}")
            return False
            
    def download_dataset(self, name, url, category="general", extract=True, metadata=None):
        """
        Download a dataset from URL
        
        Args:
            name (str): Name of the dataset
            url (str): URL to download from
            category (str): Category for organization
            extract (bool): Whether to extract the downloaded file
            metadata (dict): Additional metadata about the dataset
        """
        # Create category directory
        category_dir = self.base_dir / "raw" / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"{name}.zip"
            
        destination = category_dir / filename
        
        # Download the file
        self.logger.info(f"Starting download of {name} from {url}")
        if self.download_file(url, destination):
            
            # Extract if requested
            if extract and destination.suffix in ['.zip', '.tar', '.gz', '.bz2']:
                extract_dir = category_dir / name
                self.extract_archive(destination, extract_dir)
                
            # Save metadata
            self.save_metadata(name, url, category, metadata)
            
            return True
        return False
        
    def save_metadata(self, name, url, category, metadata=None):
        """Save metadata about downloaded dataset"""
        metadata_file = self.base_dir / "metadata" / f"{name}.json"
        
        metadata_data = {
            "name": name,
            "url": url,
            "category": category,
            "download_date": datetime.now().isoformat(),
            "file_size": (self.base_dir / "raw" / category / name).stat().st_size if (self.base_dir / "raw" / category / name).exists() else None,
            "additional_info": metadata or {}
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f, indent=2)
            
    def download_from_list(self, dataset_list):
        """
        Download multiple datasets from a list
        
        Args:
            dataset_list (list): List of dictionaries with dataset info
        """
        for dataset in dataset_list:
            self.download_dataset(
                name=dataset['name'],
                url=dataset['url'],
                category=dataset.get('category', 'general'),
                extract=dataset.get('extract', True),
                metadata=dataset.get('metadata', {})
            )
            
    def get_popular_datasets(self):
        """Return a list of popular datasets to download"""
        return [
            {
                "name": "iris",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "category": "classification",
                "extract": False,
                "metadata": {"description": "Iris flower dataset for classification"}
            },
            {
                "name": "wine_quality",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                "category": "regression",
                "extract": False,
                "metadata": {"description": "Wine quality dataset for regression"}
            },
            {
                "name": "breast_cancer",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                "category": "classification",
                "extract": False,
                "metadata": {"description": "Breast cancer Wisconsin dataset"}
            }
        ]
        
    def list_downloaded_datasets(self):
        """List all downloaded datasets"""
        metadata_dir = self.base_dir / "metadata"
        datasets = []
        
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                datasets.append(json.load(f))
                
        return datasets
        
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = self.base_dir / "temp"
        for file in temp_dir.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                import shutil
                shutil.rmtree(file)
                
        self.logger.info("Cleaned up temporary files")

def main():
    """Main function to run the dataset downloader"""
    print("🚀 Automatic Dataset Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = DatasetDownloader()
    
    while True:
        print("\nOptions:")
        print("1. Download popular datasets")
        print("2. Download custom dataset")
        print("3. List downloaded datasets")
        print("4. Clean up temporary files")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n📥 Downloading popular datasets...")
            popular_datasets = downloader.get_popular_datasets()
            downloader.download_from_list(popular_datasets)
            print("✅ Popular datasets download completed!")
            
        elif choice == "2":
            print("\n📥 Custom dataset download")
            name = input("Dataset name: ").strip()
            url = input("Download URL: ").strip()
            category = input("Category (default: general): ").strip() or "general"
            extract = input("Extract archive? (y/n, default: y): ").strip().lower() != 'n'
            
            downloader.download_dataset(name, url, category, extract)
            print("✅ Custom dataset download completed!")
            
        elif choice == "3":
            print("\n📋 Downloaded datasets:")
            datasets = downloader.list_downloaded_datasets()
            for dataset in datasets:
                print(f"- {dataset['name']} ({dataset['category']}) - {dataset['download_date']}")
                
        elif choice == "4":
            print("\n🧹 Cleaning up temporary files...")
            downloader.cleanup_temp_files()
            print("✅ Cleanup completed!")
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 