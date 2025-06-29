#!/usr/bin/env python3
"""
Example usage of the Automatic Dataset Downloader Tool
This script demonstrates how to use the downloader programmatically
"""

from enhanced_downloader import EnhancedDatasetDownloader
import json

def main():
    """Example usage of the dataset downloader"""
    
    # Initialize the downloader
    print("🚀 Initializing Dataset Downloader...")
    downloader = EnhancedDatasetDownloader()
    
    # Example 1: Download a single dataset
    print("\n📥 Example 1: Downloading a single dataset")
    success = downloader.download_dataset(
        name="example_dataset",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        category="classification",
        extract=False,
        metadata={"description": "Example dataset for demonstration"}
    )
    
    if success:
        print("✅ Dataset downloaded successfully!")
    else:
        print("❌ Failed to download dataset")
    
    # Example 2: Download multiple datasets
    print("\n📥 Example 2: Downloading multiple datasets")
    datasets_to_download = [
        {
            "name": "wine_quality",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
            "category": "regression",
            "extract": False,
            "metadata": {"description": "Wine quality dataset for regression"}
        },
        {
            "name": "diabetes",
            "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.data",
            "category": "regression",
            "extract": False,
            "metadata": {"description": "Diabetes dataset for regression"}
        }
    ]
    
    successful, total = downloader.download_from_list(datasets_to_download)
    print(f"✅ Downloaded {successful}/{total} datasets successfully!")
    
    # Example 3: List all downloaded datasets
    print("\n📋 Example 3: Listing all downloaded datasets")
    datasets = downloader.list_downloaded_datasets()
    print(f"Found {len(datasets)} downloaded datasets:")
    
    for dataset in datasets:
        size_info = f" ({dataset.get('file_size_human', 'Unknown size')})" if dataset.get('file_size_human') else ""
        print(f"- {dataset['name']} ({dataset['category']}){size_info}")
    
    # Example 4: Search for datasets
    print("\n🔍 Example 4: Searching for datasets")
    search_results = downloader.search_datasets("classification")
    print(f"Found {len(search_results)} datasets matching 'classification':")
    
    for result in search_results:
        print(f"- {result['name']} ({result['category']})")
    
    # Example 5: Check disk usage
    print("\n💾 Example 5: Checking disk usage")
    disk_usage = downloader.get_disk_usage()
    print(f"Total: {disk_usage['total']}")
    print(f"Used: {disk_usage['used']}")
    print(f"Free: {disk_usage['free']}")
    print(f"Usage: {disk_usage['usage_percent']:.1f}%")
    
    # Example 6: Clean up temporary files
    print("\n🧹 Example 6: Cleaning up temporary files")
    downloader.cleanup_temp_files()
    print("✅ Cleanup completed!")
    
    print("\n🎉 All examples completed successfully!")
    print("Check your D:/datasets/ directory to see the downloaded files.")

def create_custom_config():
    """Example of creating a custom configuration"""
    custom_config = {
        "base_directory": "D:/my_datasets",
        "download_settings": {
            "chunk_size": 16384,  # Larger chunks for faster downloads
            "timeout": 60,        # Longer timeout
            "retry_attempts": 5,  # More retry attempts
            "retry_delay": 10     # Longer delay between retries
        },
        "organization": {
            "auto_categorize": True,
            "create_date_folders": True,
            "keep_original_archives": False
        },
        "popular_datasets": [
            {
                "name": "custom_dataset_1",
                "url": "https://example.com/dataset1.zip",
                "category": "custom",
                "extract": True,
                "description": "My first custom dataset"
            },
            {
                "name": "custom_dataset_2",
                "url": "https://example.com/dataset2.csv",
                "category": "custom",
                "extract": False,
                "description": "My second custom dataset"
            }
        ]
    }
    
    # Save custom configuration
    with open("custom_config.json", "w") as f:
        json.dump(custom_config, f, indent=2)
    
    print("✅ Custom configuration saved to custom_config.json")
    print("You can use it with: downloader = EnhancedDatasetDownloader('custom_config.json')")

if __name__ == "__main__":
    print("🚀 Dataset Downloader - Example Usage")
    print("=" * 50)
    
    # Run examples
    main()
    
    # Create custom configuration example
    print("\n" + "=" * 50)
    print("Creating custom configuration example...")
    create_custom_config()
    
    print("\n👋 Example usage completed!")
    print("Check the README.md file for more detailed instructions.") 