#!/usr/bin/env python3
"""
Download 20 Different Datasets (~20MB each) for AI Agent Training
Simplified version without complex logging
"""

import requests
import json
import time
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

class SimpleDatasetDownloader:
    def __init__(self, base_dir="./ai_agent_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
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
        
    def download_file(self, url, destination, description=""):
        """Download file with progress bar"""
        try:
            print(f"    Downloading from: {url}")
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
            print(f"    ✅ Downloaded: {file_size / (1024*1024):.2f} MB")
            return True
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            return False
            
    def get_datasets(self):
        """Return list of 20 datasets across different categories"""
        return [
            # 1. Classification - Iris (small but reliable)
            {
                "name": "iris_classification",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                "category": "classification",
                "description": "Iris flower classification dataset - 150 samples, 4 features"
            },
            
            # 2. Regression - Wine Quality
            {
                "name": "wine_quality_regression",
                "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
                "category": "regression",
                "description": "Wine quality regression - 1,599 samples, 11 features"
            },
            
            # 3. Clustering - Mall Customers
            {
                "name": "mall_customers_clustering",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/mall_customers.csv",
                "category": "clustering",
                "description": "Mall customer segmentation data for clustering"
            },
            
            # 4. NLP - SMS Spam
            {
                "name": "sms_spam_nlp",
                "url": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                "category": "nlp",
                "description": "SMS spam classification dataset for NLP tasks"
            },
            
            # 5. Time Series - Air Passengers
            {
                "name": "air_passengers_timeseries",
                "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
                "category": "time_series",
                "description": "Airline passengers time series - 144 months of data"
            },
            
            # 6. Computer Vision - Image Features
            {
                "name": "image_features_cv",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv",
                "category": "computer_vision",
                "description": "Motor trend car data as image features (repurposed)"
            },
            
            # 7. Recommendation - Movie Data
            {
                "name": "movie_recommendation",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/movies.csv",
                "category": "recommendation",
                "description": "Movie data for recommendation systems"
            },
            
            # 8. Anomaly Detection - Network Data
            {
                "name": "network_anomaly",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/network.csv",
                "category": "anomaly_detection",
                "description": "Network traffic data for anomaly detection"
            },
            
            # 9. Forecasting - Sales Data
            {
                "name": "sales_forecasting",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/forecast.csv",
                "category": "forecasting",
                "description": "Sales data for time series forecasting"
            },
            
            # 10. Sentiment Analysis - Reviews
            {
                "name": "reviews_sentiment",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/amazon_reviews.csv",
                "category": "sentiment_analysis",
                "description": "Amazon reviews for sentiment analysis"
            },
            
            # 11. Fraud Detection - Credit Card
            {
                "name": "credit_fraud_detection",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/fraud.csv",
                "category": "fraud_detection",
                "description": "Credit card fraud detection dataset"
            },
            
            # 12. Customer Behavior - Shopping
            {
                "name": "customer_behavior_shopping",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/shopping.csv",
                "category": "customer_behavior",
                "description": "Customer shopping behavior data"
            },
            
            # 13. Financial - Stock Prices
            {
                "name": "stock_prices_financial",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/AAPL.csv",
                "category": "financial",
                "description": "Apple stock price time series data"
            },
            
            # 14. Healthcare - Heart Disease
            {
                "name": "heart_disease_healthcare",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/heart.csv",
                "category": "healthcare",
                "description": "Heart disease prediction dataset"
            },
            
            # 15. E-commerce - Sales
            {
                "name": "ecommerce_sales",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/sales.csv",
                "category": "ecommerce",
                "description": "E-commerce sales and customer data"
            },
            
            # 16. Social Media - Users
            {
                "name": "social_media_users",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/users.csv",
                "category": "social_media",
                "description": "Social media user behavior and demographics"
            },
            
            # 17. Weather - Climate
            {
                "name": "weather_climate",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/weather.csv",
                "category": "weather",
                "description": "Historical weather data for forecasting"
            },
            
            # 18. Traffic - Vehicles
            {
                "name": "traffic_vehicles",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/cars.csv",
                "category": "traffic",
                "description": "Vehicle traffic and performance data"
            },
            
            # 19. Energy - Power
            {
                "name": "energy_power",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/power.csv",
                "category": "energy",
                "description": "Power consumption time series data"
            },
            
            # 20. Education - Students
            {
                "name": "education_students",
                "url": "https://raw.githubusercontent.com/selva86/datasets/master/students.csv",
                "category": "education",
                "description": "Student academic performance dataset"
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
    print("📥 Downloading 20 datasets across 20 categories")
    print()
    
    downloader = SimpleDatasetDownloader()
    
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