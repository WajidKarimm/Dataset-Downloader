#!/usr/bin/env python3
"""
AI Training CSV Dataset Downloader
Downloads popular CSV datasets specifically for AI agent training
"""

import os
import sys
import requests
import json
import time
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
from tqdm import tqdm
import logging
from datetime import datetime
import shutil

class AITrainingDownloader:
    def __init__(self, base_dir="./ai_datasets"):
        """
        Initialize the AI training dataset downloader
        
        Args:
            base_dir (str): Base directory to store datasets (default: ./ai_datasets)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
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
                logging.FileHandler(log_dir / f"ai_training_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_directory_structure(self):
        """Create the directory structure for organizing datasets"""
        directories = [
            "classification",
            "regression", 
            "clustering",
            "nlp",
            "computer_vision",
            "time_series",
            "metadata",
            "logs",
            "processed"
        ]
        
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
            
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
                    
    def analyze_csv(self, file_path):
        """Analyze CSV file and create metadata"""
        try:
            df = pd.read_csv(file_path)
            metadata = {
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().astype(int).to_dict(),
                "file_size": int(file_path.stat().st_size),
                "memory_usage": int(df.memory_usage(deep=True).sum())
            }
            
            # Basic statistics for numerical columns
            numerical_cols = df.select_dtypes(include=['number']).columns
            if len(numerical_cols) > 0:
                stats_dict = df[numerical_cols].describe().to_dict()
                # Convert numpy types to native Python types
                for col in stats_dict:
                    stats_dict[col] = {k: float(v) if pd.notna(v) else None for k, v in stats_dict[col].items()}
                metadata["numerical_stats"] = stats_dict
                
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error analyzing CSV {file_path}: {str(e)}")
            return None
            
    def download_dataset(self, name, url, category="general", description=""):
        """
        Download a dataset and analyze it
        
        Args:
            name (str): Name of the dataset
            url (str): URL to download from
            category (str): Category for organization
            description (str): Description of the dataset
        """
        # Create category directory
        category_dir = self.base_dir / category
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
            
            # Analyze the CSV file
            analysis = self.analyze_csv(destination)
            if analysis:
                self.logger.info(f"Successfully analyzed {name}")
                
            # Save metadata
            self.save_metadata(name, url, category, description, analysis)
            
            return True
        return False
        
    def save_metadata(self, name, url, category, description, analysis=None):
        """Save metadata about downloaded dataset"""
        metadata_file = self.base_dir / "metadata" / f"{name}.json"
        
        metadata_data = {
            "name": name,
            "url": url,
            "category": category,
            "description": description,
            "download_date": datetime.now().isoformat(),
            "analysis": analysis
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata_data, f, indent=2)
            
    def get_ai_training_datasets(self):
        """Return a list of popular AI training datasets"""
        return [
            # Classification datasets
            {
                "name": "iris_classification",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "category": "classification",
                "description": "Iris flower classification dataset - 150 samples, 4 features"
            },
            {
                "name": "breast_cancer_classification",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
                "category": "classification",
                "description": "Breast cancer Wisconsin dataset - 569 samples, 30 features"
            },
            {
                "name": "adult_income_classification",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                "category": "classification",
                "description": "Adult income classification - 48,842 samples, 14 features"
            },
            
            # Regression datasets
            {
                "name": "wine_quality_regression",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                "category": "regression",
                "description": "Wine quality regression - 1,599 samples, 11 features"
            },
            {
                "name": "housing_regression",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
                "category": "regression",
                "description": "Boston housing prices - 506 samples, 13 features"
            },
            
            # Clustering datasets
            {
                "name": "customer_segmentation",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv",
                "category": "clustering",
                "description": "Motor trend car data for clustering - 32 samples, 11 features"
            },
            
            # Time series datasets
            {
                "name": "air_passengers",
                "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
                "category": "time_series",
                "description": "Airline passengers time series - 144 months of data"
            },
            
            # NLP datasets
            {
                "name": "spam_classification",
                "url": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                "category": "nlp",
                "description": "SMS spam classification dataset"
            }
        ]
        
    def download_all_ai_datasets(self):
        """Download all AI training datasets"""
        datasets = self.get_ai_training_datasets()
        successful = 0
        
        print(f"📥 Downloading {len(datasets)} AI training datasets...")
        print("=" * 60)
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n[{i}/{len(datasets)}] Downloading {dataset['name']}...")
            print(f"   Category: {dataset['category']}")
            print(f"   Description: {dataset['description']}")
            
            if self.download_dataset(
                name=dataset['name'],
                url=dataset['url'],
                category=dataset['category'],
                description=dataset['description']
            ):
                successful += 1
                print(f"   ✅ Success!")
            else:
                print(f"   ❌ Failed!")
                
        return successful, len(datasets)
        
    def list_downloaded_datasets(self):
        """List all downloaded datasets with analysis"""
        metadata_dir = self.base_dir / "metadata"
        if not metadata_dir.exists():
            print("No datasets found.")
            return
            
        datasets = []
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                dataset = json.load(f)
                datasets.append(dataset)
                
        print(f"\n📋 Found {len(datasets)} downloaded datasets:")
        print("=" * 60)
        
        for dataset in datasets:
            analysis = dataset.get('analysis', {})
            print(f"\n📁 {dataset['name']} ({dataset['category']})")
            print(f"   📝 {dataset['description']}")
            print(f"   📅 Downloaded: {dataset['download_date'][:10]}")
            
            if analysis:
                print(f"   📊 Rows: {analysis.get('rows', 'N/A')}")
                print(f"   📊 Columns: {analysis.get('columns', 'N/A')}")
                print(f"   💾 Size: {analysis.get('file_size', 'N/A')} bytes")
                
    def get_dataset_summary(self):
        """Get summary statistics of downloaded datasets"""
        metadata_dir = self.base_dir / "metadata"
        if not metadata_dir.exists():
            return
            
        datasets = []
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                dataset = json.load(f)
                datasets.append(dataset)
                
        if not datasets:
            return
            
        # Group by category
        categories = {}
        for dataset in datasets:
            cat = dataset['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(dataset)
            
        print("\n📊 Dataset Summary:")
        print("=" * 40)
        print(f"Total datasets: {len(datasets)}")
        print(f"Categories: {len(categories)}")
        
        for category, cat_datasets in categories.items():
            print(f"\n📁 {category.upper()}: {len(cat_datasets)} datasets")
            for dataset in cat_datasets:
                analysis = dataset.get('analysis', {})
                rows = analysis.get('rows', 'N/A')
                cols = analysis.get('columns', 'N/A')
                print(f"   - {dataset['name']}: {rows} rows × {cols} columns")

def main():
    """Main function to run the AI training dataset downloader"""
    print("🤖 AI Training CSV Dataset Downloader")
    print("=" * 60)
    print("📥 Downloads popular datasets for AI/ML training")
    print()
    
    # Initialize downloader
    downloader = AITrainingDownloader()
    
    while True:
        print("\nOptions:")
        print("1. Download all AI training datasets")
        print("2. Download specific category")
        print("3. List downloaded datasets")
        print("4. Show dataset summary")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🤖 Downloading AI training datasets...")
            successful, total = downloader.download_all_ai_datasets()
            print(f"\n🎉 Download completed: {successful}/{total} datasets successful!")
            print(f"📁 Datasets saved to: {downloader.base_dir}")
            
        elif choice == "2":
            print("\n📁 Available categories:")
            categories = ["classification", "regression", "clustering", "nlp", "time_series"]
            for i, cat in enumerate(categories, 1):
                print(f"{i}. {cat}")
            
            try:
                cat_choice = int(input("Select category (1-5): ")) - 1
                if 0 <= cat_choice < len(categories):
                    category = categories[cat_choice]
                    datasets = [d for d in downloader.get_ai_training_datasets() if d['category'] == category]
                    
                    print(f"\n📥 Downloading {category} datasets...")
                    successful = 0
                    for dataset in datasets:
                        if downloader.download_dataset(
                            name=dataset['name'],
                            url=dataset['url'],
                            category=dataset['category'],
                            description=dataset['description']
                        ):
                            successful += 1
                    
                    print(f"✅ Downloaded {successful}/{len(datasets)} {category} datasets!")
                else:
                    print("❌ Invalid choice.")
            except ValueError:
                print("❌ Please enter a valid number.")
                
        elif choice == "3":
            downloader.list_downloaded_datasets()
            
        elif choice == "4":
            downloader.get_dataset_summary()
            
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 