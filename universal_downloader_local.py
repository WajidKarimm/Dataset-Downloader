#!/usr/bin/env python3
"""
Universal Dataset Downloader Tool (Local Version)
Downloads datasets in multiple formats (CSV, Excel, JSON, ARFF, XML, TSV, TXT)
with intelligent parsing and automatic organization to local directory
"""

import os
import sys
import requests
import zipfile
import tarfile
import json
import time
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import logging
from datetime import datetime
import shutil
import csv
import io

class UniversalDatasetDownloaderLocal:
    def __init__(self, base_dir="./datasets"):
        """
        Initialize the universal dataset downloader (local version)
        
        Args:
            base_dir (str): Base directory to store datasets (default: ./datasets)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Create necessary directories
        self.create_directory_structure()
        
        # Supported file formats
        self.supported_formats = {
            'csv': self.parse_csv,
            'xlsx': self.parse_excel,
            'xls': self.parse_excel,
            'json': self.parse_json,
            'arff': self.parse_arff,
            'xml': self.parse_xml,
            'tsv': self.parse_tsv,
            'txt': self.parse_txt,
            'zip': self.extract_archive,
            'tar': self.extract_archive,
            'gz': self.extract_archive
        }
        
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.base_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"universal_downloader_{datetime.now().strftime('%Y%m%d')}.log"),
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
            "backup",
            "formats"
        ]
        
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
            
        # Create format-specific directories
        format_dirs = ["csv", "excel", "json", "arff", "xml", "tsv", "txt"]
        for format_dir in format_dirs:
            (self.base_dir / "formats" / format_dir).mkdir(exist_ok=True)
            
    def download_file_with_retry(self, url, destination, max_retries=3):
        """
        Download a file from URL with retry mechanism and progress bar
        
        Args:
            url (str): URL to download from
            destination (Path): Destination file path
            max_retries (int): Maximum number of retry attempts
        """
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(destination, 'wb') as file:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                             desc=f"{destination.name} (Attempt {attempt + 1})") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                file.write(chunk)
                                pbar.update(len(chunk))
                                
                self.logger.info(f"Successfully downloaded: {destination}")
                return True
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
                if attempt < max_retries:
                    self.logger.info(f"Retrying in 5 seconds...")
                    time.sleep(5)
                else:
                    self.logger.error(f"All attempts failed for {url}")
                    return False
                    
    def detect_file_format(self, file_path):
        """Detect the file format based on extension and content"""
        extension = file_path.suffix.lower().lstrip('.')
        
        # Check if it's a supported format
        if extension in self.supported_formats:
            return extension
            
        # Try to detect format from content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                first_line = f.readline().strip()
                
                # Detect JSON
                if first_line.startswith('{') or first_line.startswith('['):
                    return 'json'
                    
                # Detect CSV/TSV
                if ',' in first_line:
                    return 'csv'
                elif '\t' in first_line:
                    return 'tsv'
                    
                # Detect ARFF
                if first_line.lower().startswith('@relation'):
                    return 'arff'
                    
                # Detect XML
                if first_line.strip().startswith('<?xml') or first_line.strip().startswith('<'):
                    return 'xml'
                    
        except Exception:
            pass
            
        return 'txt'  # Default to text
        
    def parse_csv(self, file_path, output_dir):
        """Parse CSV file and create metadata"""
        try:
            df = pd.read_csv(file_path)
            metadata = {
                "format": "csv",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": str(df.dtypes.to_dict()),
                "missing_values": df.isnull().sum().to_dict(),
                "file_size": file_path.stat().st_size
            }
            
            # Save parsed info
            info_file = output_dir / f"{file_path.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing CSV {file_path}: {str(e)}")
            return None
            
    def parse_excel(self, file_path, output_dir):
        """Parse Excel file and create metadata"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(file_path)
            sheets = {}
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheets[sheet_name] = {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": str(df.dtypes.to_dict()),
                    "missing_values": df.isnull().sum().to_dict()
                }
                
            metadata = {
                "format": "excel",
                "sheets": sheets,
                "sheet_names": excel_file.sheet_names,
                "file_size": file_path.stat().st_size
            }
            
            # Save parsed info
            info_file = output_dir / f"{file_path.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing Excel {file_path}: {str(e)}")
            return None
            
    def parse_json(self, file_path, output_dir):
        """Parse JSON file and create metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            metadata = {
                "format": "json",
                "data_type": type(data).__name__,
                "file_size": file_path.stat().st_size
            }
            
            if isinstance(data, list):
                metadata["items"] = len(data)
                if data and isinstance(data[0], dict):
                    metadata["keys"] = list(data[0].keys())
            elif isinstance(data, dict):
                metadata["keys"] = list(data.keys())
                
            # Save parsed info
            info_file = output_dir / f"{file_path.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing JSON {file_path}: {str(e)}")
            return None
            
    def parse_arff(self, file_path, output_dir):
        """Parse ARFF file and create metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            lines = content.split('\n')
            attributes = []
            data_start = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if line.lower().startswith('@attribute'):
                    attributes.append(line)
                elif line.lower() == '@data':
                    data_start = i + 1
                    break
                    
            metadata = {
                "format": "arff",
                "attributes": attributes,
                "attribute_count": len(attributes),
                "file_size": file_path.stat().st_size
            }
            
            # Save parsed info
            info_file = output_dir / f"{file_path.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing ARFF {file_path}: {str(e)}")
            return None
            
    def parse_xml(self, file_path, output_dir):
        """Parse XML file and create metadata"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            metadata = {
                "format": "xml",
                "root_tag": root.tag,
                "file_size": file_path.stat().st_size
            }
            
            # Count elements
            elements = list(root.iter())
            metadata["total_elements"] = len(elements)
            metadata["unique_tags"] = list(set(elem.tag for elem in elements))
            
            # Save parsed info
            info_file = output_dir / f"{file_path.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing XML {file_path}: {str(e)}")
            return None
            
    def parse_tsv(self, file_path, output_dir):
        """Parse TSV file and create metadata"""
        try:
            df = pd.read_csv(file_path, sep='\t')
            metadata = {
                "format": "tsv",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "data_types": str(df.dtypes.to_dict()),
                "missing_values": df.isnull().sum().to_dict(),
                "file_size": file_path.stat().st_size
            }
            
            # Save parsed info
            info_file = output_dir / f"{file_path.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing TSV {file_path}: {str(e)}")
            return None
            
    def parse_txt(self, file_path, output_dir):
        """Parse TXT file and create metadata"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            lines = content.split('\n')
            words = content.split()
            
            metadata = {
                "format": "txt",
                "lines": len(lines),
                "words": len(words),
                "characters": len(content),
                "file_size": file_path.stat().st_size
            }
            
            # Save parsed info
            info_file = output_dir / f"{file_path.stem}_info.json"
            with open(info_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error parsing TXT {file_path}: {str(e)}")
            return None
            
    def extract_archive(self, file_path, output_dir):
        """Extract archive files"""
        try:
            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
            elif file_path.suffix in ['.tar', '.gz', '.bz2']:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(output_dir)
                    
            self.logger.info(f"Successfully extracted: {file_path} to {output_dir}")
            return {"format": "archive", "extracted": True}
            
        except Exception as e:
            self.logger.error(f"Error extracting {file_path}: {str(e)}")
            return None
            
    def download_and_parse_dataset(self, name, url, category="general", metadata=None):
        """
        Download a dataset and automatically parse it based on format
        
        Args:
            name (str): Name of the dataset
            url (str): URL to download from
            category (str): Category for organization
            metadata (dict): Additional metadata about the dataset
        """
        # Create category directory
        category_dir = self.base_dir / "raw" / category
        category_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension from URL
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = f"{name}.csv"
            
        destination = category_dir / filename
        
        # Download the file
        self.logger.info(f"Starting download of {name} from {url}")
        if self.download_file_with_retry(url, destination):
            
            # Detect and parse file format
            file_format = self.detect_file_format(destination)
            self.logger.info(f"Detected format: {file_format}")
            
            # Parse the file
            format_dir = self.base_dir / "formats" / file_format
            format_dir.mkdir(exist_ok=True)
            
            if file_format in self.supported_formats:
                parse_result = self.supported_formats[file_format](destination, format_dir)
                if parse_result:
                    self.logger.info(f"Successfully parsed {file_format} file")
                    
            # Save metadata
            self.save_metadata(name, url, category, file_format, metadata, parse_result)
            
            return True
        return False
        
    def save_metadata(self, name, url, category, file_format, metadata=None, parse_result=None):
        """Save metadata about downloaded dataset"""
        metadata_file = self.base_dir / "metadata" / f"{name}.json"
        
        metadata_data = {
            "name": name,
            "url": url,
            "category": category,
            "file_format": file_format,
            "download_date": datetime.now().isoformat(),
            "file_size": (self.base_dir / "raw" / category / name).stat().st_size if (self.base_dir / "raw" / category / name).exists() else None,
            "additional_info": metadata or {},
            "parse_result": parse_result
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f, indent=2)
            
    def get_universal_datasets(self):
        """Return a list of datasets in various formats"""
        return [
            {
                "name": "iris_csv",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "category": "classification",
                "description": "Iris dataset in CSV format"
            },
            {
                "name": "wine_quality_csv",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                "category": "regression",
                "description": "Wine quality dataset in CSV format"
            },
            {
                "name": "sample_json",
                "url": "https://jsonplaceholder.typicode.com/posts",
                "category": "api_data",
                "description": "Sample JSON data from API"
            },
            {
                "name": "sample_xml",
                "url": "https://www.w3schools.com/xml/note.xml",
                "category": "sample",
                "description": "Sample XML data"
            },
            {
                "name": "sample_tsv",
                "url": "https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.tsv",
                "category": "economics",
                "description": "GDP data in TSV format"
            }
        ]
        
    def download_universal_datasets(self):
        """Download datasets in various formats"""
        datasets = self.get_universal_datasets()
        successful = 0
        
        for dataset in datasets:
            self.logger.info(f"Downloading {dataset['name']}...")
            if self.download_and_parse_dataset(
                name=dataset['name'],
                url=dataset['url'],
                category=dataset['category'],
                metadata={"description": dataset['description']}
            ):
                successful += 1
                
        return successful, len(datasets)

def main():
    """Main function to run the universal dataset downloader"""
    print("🌐 Universal Dataset Downloader (Local Version)")
    print("=" * 60)
    print("📄 Supports: CSV, Excel, JSON, ARFF, XML, TSV, TXT")
    print()
    
    # Initialize downloader
    downloader = UniversalDatasetDownloaderLocal()
    
    while True:
        print("\nOptions:")
        print("1. Download universal datasets (various formats)")
        print("2. Download custom dataset")
        print("3. List downloaded datasets")
        print("4. Show supported formats")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n📥 Downloading datasets in various formats...")
            successful, total = downloader.download_universal_datasets()
            print(f"✅ Download completed: {successful}/{total} datasets successful!")
            
        elif choice == "2":
            print("\n📥 Custom dataset download")
            name = input("Dataset name: ").strip()
            url = input("Download URL: ").strip()
            category = input("Category (default: general): ").strip() or "general"
            description = input("Description (optional): ").strip()
            
            metadata = {"description": description} if description else {}
            downloader.download_and_parse_dataset(name, url, category, metadata)
            print("✅ Custom dataset download completed!")
            
        elif choice == "3":
            print("\n📋 Downloaded datasets:")
            metadata_dir = downloader.base_dir / "metadata"
            if metadata_dir.exists():
                for metadata_file in metadata_dir.glob("*.json"):
                    with open(metadata_file, 'r') as f:
                        dataset = json.load(f)
                    print(f"- {dataset['name']} ({dataset['file_format']}) - {dataset['category']}")
            else:
                print("No datasets found.")
                
        elif choice == "4":
            print("\n📄 Supported Formats:")
            for format_name, parser in downloader.supported_formats.items():
                print(f"- {format_name.upper()}")
                
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 