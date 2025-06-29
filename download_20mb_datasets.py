#!/usr/bin/env python3
"""
Download 20 Different Datasets (~20MB each) for AI Agent Training
Various categories: Classification, Regression, NLP, Time Series, etc.
"""

import requests
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime
import zipfile
import io

class DatasetDownloader:
    def __init__(self, base_dir="./ai_agent_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Create category directories
        self.categories = [
            "classification", "regression", "clustering", "nlp", 
            "time_series", "computer_vision", "recommendation", "anomaly_detection",
            "forecasting", "sentiment_analysis", "fraud_detection", "customer_behavior",
            "financial", "healthcare", "ecommerce", "social_media",
            "weather", "traffic", "energy", "education"
        ]
        
        for category in self.categories:
            (self.base_dir / category).mkdir(exist_ok=True)
            
        (self.base_dir / "metadata").mkdir(exist_ok=True)
        (self.base_dir / "logs").mkdir(exist_ok=True)
        
    def setup_logging(self):
        log_dir = self.base_dir / "logs"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"download_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def download_file(self, url, destination, description=""):
        """Download file with progress bar"""
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
            file_size = destination.stat().st_size
            self.logger.info(f"Downloaded {destination.name}: {file_size / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {url}: {str(e)}")
            return False
            
    def get_datasets(self):
        """Return list of 20 datasets across different categories"""
        return [
            # 1. Classification - Credit Card Fraud Detection
            {
                "name": "credit_card_fraud",
                "url": "https://raw.githubusercontent.com/datasets/credit-card-fraud/master/data/creditcard.csv",
                "category": "fraud_detection",
                "description": "Credit card fraud detection dataset with 284,807 transactions"
            },
            
            # 2. Regression - House Prices
            {
                "name": "house_prices_advanced",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/Housing.csv",
                "category": "financial",
                "description": "Advanced house pricing dataset with multiple features"
            },
            
            # 3. NLP - Amazon Reviews
            {
                "name": "amazon_reviews",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/amazon_reviews.csv",
                "category": "sentiment_analysis",
                "description": "Amazon product reviews for sentiment analysis"
            },
            
            # 4. Time Series - Stock Prices
            {
                "name": "stock_prices",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/AAPL.csv",
                "category": "financial",
                "description": "Apple stock price time series data"
            },
            
            # 5. Clustering - Customer Segmentation
            {
                "name": "customer_segmentation_large",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/mall_customers.csv",
                "category": "customer_behavior",
                "description": "Mall customer segmentation data"
            },
            
            # 6. Healthcare - Heart Disease
            {
                "name": "heart_disease",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/heart.csv",
                "category": "healthcare",
                "description": "Heart disease prediction dataset"
            },
            
            # 7. E-commerce - Sales Data
            {
                "name": "ecommerce_sales",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/sales.csv",
                "category": "ecommerce",
                "description": "E-commerce sales and customer data"
            },
            
            # 8. Weather - Climate Data
            {
                "name": "weather_data",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/weather.csv",
                "category": "weather",
                "description": "Historical weather data for forecasting"
            },
            
            # 9. Traffic - Vehicle Data
            {
                "name": "traffic_analysis",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/cars.csv",
                "category": "traffic",
                "description": "Vehicle traffic and performance data"
            },
            
            # 10. Energy - Power Consumption
            {
                "name": "energy_consumption",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/power.csv",
                "category": "energy",
                "description": "Power consumption time series data"
            },
            
            # 11. Education - Student Performance
            {
                "name": "student_performance",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/students.csv",
                "category": "education",
                "description": "Student academic performance dataset"
            },
            
            # 12. Social Media - User Behavior
            {
                "name": "social_media_users",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/users.csv",
                "category": "social_media",
                "description": "Social media user behavior and demographics"
            },
            
            # 13. Recommendation - Movie Ratings
            {
                "name": "movie_ratings",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/movies.csv",
                "category": "recommendation",
                "description": "Movie ratings and user preferences"
            },
            
            # 14. Anomaly Detection - Network Traffic
            {
                "name": "network_traffic",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/network.csv",
                "category": "anomaly_detection",
                "description": "Network traffic data for anomaly detection"
            },
            
            # 15. Forecasting - Sales Prediction
            {
                "name": "sales_forecasting",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/forecast.csv",
                "category": "forecasting",
                "description": "Sales data for time series forecasting"
            },
            
            # 16. Computer Vision - Image Metadata
            {
                "name": "image_metadata",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/images.csv",
                "category": "computer_vision",
                "description": "Image metadata and features for CV tasks"
            },
            
            # 17. Healthcare - Medical Records
            {
                "name": "medical_records",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/medical.csv",
                "category": "healthcare",
                "description": "Medical records and patient data"
            },
            
            # 18. Financial - Market Data
            {
                "name": "market_data",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/market.csv",
                "category": "financial",
                "description": "Financial market indicators and trends"
            },
            
            # 19. NLP - Text Classification
            {
                "name": "text_classification",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/texts.csv",
                "category": "nlp",
                "description": "Text data for classification tasks"
            },
            
            # 20. Time Series - Sensor Data
            {
                "name": "sensor_data",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/sensors.csv",
                "category": "time_series",
                "description": "IoT sensor time series data"
            }
        ]
        
    def download_all_datasets(self):
        """Download all 20 datasets"""
        datasets = self.get_datasets()
        successful = 0
        
        print(f"🚀 Downloading 20 datasets across {len(self.categories)} categories...")
        print("=" * 70)
        
        for i, dataset in enumerate(datasets, 1):
            print(f"\n[{i:2d}/20] 📥 {dataset['name']}")
            print(f"    Category: {dataset['category']}")
            print(f"    Description: {dataset['description']}")
            
            # Create destination path
            dest_file = self.base_dir / dataset['category'] / f"{dataset['name']}.csv"
            
            if self.download_file(dataset['url'], dest_file, dataset['description']):
                successful += 1
                
                # Save metadata
                metadata = {
                    "name": dataset['name'],
                    "category": dataset['category'],
                    "description": dataset['description'],
                    "url": dataset['url'],
                    "download_date": datetime.now().isoformat(),
                    "file_size_mb": dest_file.stat().st_size / (1024*1024)
                }
                
                metadata_file = self.base_dir / "metadata" / f"{dataset['name']}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                    
                print(f"    ✅ Success! ({metadata['file_size_mb']:.2f} MB)")
            else:
                print(f"    ❌ Failed!")
                
        return successful, len(datasets)
        
    def create_summary(self):
        """Create a summary of downloaded datasets"""
        metadata_dir = self.base_dir / "metadata"
        if not metadata_dir.exists():
            return
            
        datasets = []
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                datasets.append(json.load(f))
                
        if not datasets:
            return
            
        # Group by category
        categories = {}
        total_size = 0
        for dataset in datasets:
            cat = dataset['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(dataset)
            total_size += dataset.get('file_size_mb', 0)
            
        print(f"\n📊 Download Summary:")
        print("=" * 50)
        print(f"Total datasets: {len(datasets)}")
        print(f"Total size: {total_size:.2f} MB")
        print(f"Categories: {len(categories)}")
        
        for category, cat_datasets in categories.items():
            cat_size = sum(d.get('file_size_mb', 0) for d in cat_datasets)
            print(f"\n📁 {category.upper()}: {len(cat_datasets)} datasets ({cat_size:.2f} MB)")
            for dataset in cat_datasets:
                print(f"   - {dataset['name']}: {dataset.get('file_size_mb', 0):.2f} MB")

def main():
    print("🤖 AI Agent Training Dataset Downloader")
    print("=" * 60)
    print("📥 Downloading 20 datasets (~20MB each) across 20 categories")
    print()
    
    downloader = DatasetDownloader()
    
    # Download all datasets
    successful, total = downloader.download_all_datasets()
    
    print(f"\n🎉 Download completed: {successful}/{total} datasets successful!")
    print(f"📁 Datasets saved to: {downloader.base_dir}")
    
    # Create summary
    downloader.create_summary()
    
    print(f"\n📝 Next steps:")
    print(f"1. Check the 'ai_agent_datasets' folder for your datasets")
    print(f"2. Each category has its own subfolder")
    print(f"3. Metadata files contain dataset information")
    print(f"4. Ready for AI agent training!")

if __name__ == "__main__":
    main() 