#!/usr/bin/env python3
"""
Enhanced Automatic Dataset Downloader Tool
Downloads datasets from various sources to D drive automatically
with configuration support and advanced features
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
import shutil

class EnhancedDatasetDownloader:
    def __init__(self, config_file="config.json"):
        """
        Initialize the enhanced dataset downloader
        
        Args:
            config_file (str): Path to configuration file
        """
        self.config = self.load_config(config_file)
        self.base_dir = Path(self.config.get("base_directory", "D:/datasets"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Create necessary directories
        self.create_directory_structure()
        
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Configuration file {config_file} not found. Using defaults.")
            return {}
        except json.JSONDecodeError:
            print(f"Error parsing configuration file {config_file}. Using defaults.")
            return {}
        
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
            "logs",
            "backup"
        ]
        
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
            
    def download_file_with_retry(self, url, destination, max_retries=None):
        """
        Download a file from URL with retry mechanism and progress bar
        
        Args:
            url (str): URL to download from
            destination (Path): Destination file path
            max_retries (int): Maximum number of retry attempts
        """
        if max_retries is None:
            max_retries = self.config.get("download_settings", {}).get("retry_attempts", 3)
            
        chunk_size = self.config.get("download_settings", {}).get("chunk_size", 8192)
        timeout = self.config.get("download_settings", {}).get("timeout", 30)
        retry_delay = self.config.get("download_settings", {}).get("retry_delay", 5)
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=timeout)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(destination, 'wb') as file:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"{destination.name} (Attempt {attempt + 1})") as pbar:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))
                                
                self.logger.info(f"Successfully downloaded: {destination}")
                return True
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries:
                    self.logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    self.logger.error(f"All attempts failed for {url}")
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
        
        # Check if file already exists
        if destination.exists():
            self.logger.info(f"File already exists: {destination}")
            response = input("Do you want to overwrite? (y/n): ").lower()
            if response != 'y':
                self.logger.info("Skipping download.")
                return True
        
        # Download the file
        self.logger.info(f"Starting download of {name} from {url}")
        if self.download_file_with_retry(url, destination):
            
            # Extract if requested
            if extract and destination.suffix in ['.zip', '.tar', '.gz', '.bz2']:
                extract_dir = category_dir / name
                self.extract_archive(destination, extract_dir)
                
            # Save metadata
            self.save_metadata(name, url, category, metadata)
            
            # Backup if configured
            if self.config.get("organization", {}).get("keep_original_archives", True):
                self.backup_file(destination)
            
            return True
        return False
        
    def backup_file(self, file_path):
        """Create a backup of the downloaded file"""
        backup_dir = self.base_dir / "backup"
        backup_dir.mkdir(exist_ok=True)
        
        backup_path = backup_dir / file_path.name
        shutil.copy2(file_path, backup_path)
        self.logger.info(f"Backup created: {backup_path}")
        
    def save_metadata(self, name, url, category, metadata=None):
        """Save metadata about downloaded dataset"""
        metadata_file = self.base_dir / "metadata" / f"{name}.json"
        
        # Calculate file size
        file_size = None
        raw_file = self.base_dir / "raw" / category / name
        if raw_file.exists():
            if raw_file.is_file():
                file_size = raw_file.stat().st_size
            else:
                file_size = sum(f.stat().st_size for f in raw_file.rglob('*') if f.is_file())
        
        metadata_data = {
            "name": name,
            "url": url,
            "category": category,
            "download_date": datetime.now().isoformat(),
            "file_size": file_size,
            "file_size_human": self.human_readable_size(file_size) if file_size else None,
            "additional_info": metadata or {}
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f, indent=2)
            
    def human_readable_size(self, size_bytes):
        """Convert bytes to human readable format"""
        if size_bytes is None:
            return None
            
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
        
    def download_from_list(self, dataset_list):
        """
        Download multiple datasets from a list
        
        Args:
            dataset_list (list): List of dictionaries with dataset info
        """
        total = len(dataset_list)
        successful = 0
        
        for i, dataset in enumerate(dataset_list, 1):
            self.logger.info(f"Downloading dataset {i}/{total}: {dataset['name']}")
            
            if self.download_dataset(
                name=dataset['name'],
                url=dataset['url'],
                category=dataset.get('category', 'general'),
                extract=dataset.get('extract', True),
                metadata=dataset.get('metadata', {})
            ):
                successful += 1
                
        self.logger.info(f"Download completed: {successful}/{total} datasets successful")
        return successful, total
        
    def get_popular_datasets(self):
        """Return a list of popular datasets from configuration"""
        return self.config.get("popular_datasets", [])
        
    def list_downloaded_datasets(self):
        """List all downloaded datasets with detailed information"""
        metadata_dir = self.base_dir / "metadata"
        datasets = []
        
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                datasets.append(json.load(f))
                
        return sorted(datasets, key=lambda x: x['download_date'], reverse=True)
        
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        temp_dir = self.base_dir / "temp"
        for file in temp_dir.glob("*"):
            if file.is_file():
                file.unlink()
            elif file.is_dir():
                shutil.rmtree(file)
                
        self.logger.info("Cleaned up temporary files")
        
    def get_disk_usage(self):
        """Get disk usage information"""
        total, used, free = shutil.disk_usage(self.base_dir)
        return {
            "total": self.human_readable_size(total),
            "used": self.human_readable_size(used),
            "free": self.human_readable_size(free),
            "usage_percent": (used / total) * 100
        }
        
    def search_datasets(self, query):
        """Search through downloaded datasets"""
        datasets = self.list_downloaded_datasets()
        results = []
        
        query = query.lower()
        for dataset in datasets:
            if (query in dataset['name'].lower() or 
                query in dataset['category'].lower() or
                query in dataset.get('additional_info', {}).get('description', '').lower()):
                results.append(dataset)
                
        return results

def main():
    """Main function to run the enhanced dataset downloader"""
    print("🚀 Enhanced Automatic Dataset Downloader")
    print("=" * 50)
    
    # Initialize downloader
    downloader = EnhancedDatasetDownloader()
    
    # Show disk usage
    disk_usage = downloader.get_disk_usage()
    print(f"💾 Disk Usage: {disk_usage['used']} / {disk_usage['total']} ({disk_usage['usage_percent']:.1f}%)")
    print(f"📁 Base Directory: {downloader.base_dir}")
    print()
    
    while True:
        print("\nOptions:")
        print("1. Download popular datasets")
        print("2. Download custom dataset")
        print("3. List downloaded datasets")
        print("4. Search datasets")
        print("5. Show disk usage")
        print("6. Clean up temporary files")
        print("7. Exit")
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            print("\n📥 Downloading popular datasets...")
            popular_datasets = downloader.get_popular_datasets()
            if popular_datasets:
                successful, total = downloader.download_from_list(popular_datasets)
                print(f"✅ Download completed: {successful}/{total} datasets successful!")
            else:
                print("❌ No popular datasets configured.")
                
        elif choice == "2":
            print("\n📥 Custom dataset download")
            name = input("Dataset name: ").strip()
            url = input("Download URL: ").strip()
            category = input("Category (default: general): ").strip() or "general"
            extract = input("Extract archive? (y/n, default: y): ").strip().lower() != 'n'
            description = input("Description (optional): ").strip()
            
            metadata = {"description": description} if description else {}
            downloader.download_dataset(name, url, category, extract, metadata)
            print("✅ Custom dataset download completed!")
            
        elif choice == "3":
            print("\n📋 Downloaded datasets:")
            datasets = downloader.list_downloaded_datasets()
            for dataset in datasets:
                size_info = f" ({dataset.get('file_size_human', 'Unknown size')})" if dataset.get('file_size_human') else ""
                print(f"- {dataset['name']} ({dataset['category']}){size_info}")
                print(f"  Downloaded: {dataset['download_date'][:10]}")
                if dataset.get('additional_info', {}).get('description'):
                    print(f"  Description: {dataset['additional_info']['description']}")
                print()
                
        elif choice == "4":
            print("\n🔍 Search datasets")
            query = input("Enter search term: ").strip()
            results = downloader.search_datasets(query)
            if results:
                print(f"\nFound {len(results)} matching datasets:")
                for dataset in results:
                    print(f"- {dataset['name']} ({dataset['category']})")
            else:
                print("No datasets found matching your search.")
                
        elif choice == "5":
            print("\n💾 Disk Usage Information:")
            disk_usage = downloader.get_disk_usage()
            print(f"Total: {disk_usage['total']}")
            print(f"Used: {disk_usage['used']}")
            print(f"Free: {disk_usage['free']}")
            print(f"Usage: {disk_usage['usage_percent']:.1f}%")
            
        elif choice == "6":
            print("\n🧹 Cleaning up temporary files...")
            downloader.cleanup_temp_files()
            print("✅ Cleanup completed!")
            
        elif choice == "7":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 