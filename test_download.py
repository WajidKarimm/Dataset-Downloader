#!/usr/bin/env python3
"""
Test script for the Automatic Dataset Downloader Tool
This script will automatically download sample datasets to test functionality
"""

from enhanced_downloader import EnhancedDatasetDownloader
import time

def test_download():
    """Test the dataset downloader functionality"""
    print("🧪 Testing Automatic Dataset Downloader Tool")
    print("=" * 50)
    
    # Initialize the downloader
    print("📁 Initializing downloader...")
    downloader = EnhancedDatasetDownloader()
    
    # Show initial disk usage
    disk_usage = downloader.get_disk_usage()
    print(f"💾 Initial Disk Usage: {disk_usage['used']} / {disk_usage['total']} ({disk_usage['usage_percent']:.1f}%)")
    print(f"📁 Base Directory: {downloader.base_dir}")
    print()
    
    # Test dataset 1: Iris dataset (small, fast download)
    print("📥 Test 1: Downloading Iris dataset...")
    success1 = downloader.download_dataset(
        name="iris_test",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        category="classification",
        extract=False,
        metadata={"description": "Iris flower dataset for testing", "test": True},
        overwrite=True
    )
    
    if success1:
        print("✅ Iris dataset downloaded successfully!")
    else:
        print("❌ Failed to download Iris dataset")
    
    print()
    
    # Test dataset 2: Wine quality dataset
    print("📥 Test 2: Downloading Wine Quality dataset...")
    success2 = downloader.download_dataset(
        name="wine_quality_test",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv",
        category="regression",
        extract=False,
        metadata={"description": "Wine quality dataset for testing", "test": True},
        overwrite=True
    )
    
    if success2:
        print("✅ Wine Quality dataset downloaded successfully!")
    else:
        print("❌ Failed to download Wine Quality dataset")
    
    print()
    
    # Test dataset 3: Diabetes dataset
    print("📥 Test 3: Downloading Diabetes dataset...")
    success3 = downloader.download_dataset(
        name="diabetes_test",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/diabetes/diabetes.data",
        category="regression",
        extract=False,
        metadata={"description": "Diabetes dataset for testing", "test": True},
        overwrite=True
    )
    
    if success3:
        print("✅ Diabetes dataset downloaded successfully!")
    else:
        print("❌ Failed to download Diabetes dataset")
    
    print()
    
    # List downloaded datasets
    print("📋 Listing downloaded datasets...")
    datasets = downloader.list_downloaded_datasets()
    print(f"Found {len(datasets)} downloaded datasets:")
    
    for dataset in datasets:
        size_info = f" ({dataset.get('file_size_human', 'Unknown size')})" if dataset.get('file_size_human') else ""
        print(f"- {dataset['name']} ({dataset['category']}){size_info}")
        if dataset.get('additional_info', {}).get('description'):
            print(f"  Description: {dataset['additional_info']['description']}")
    
    print()
    
    # Test search functionality
    print("🔍 Testing search functionality...")
    search_results = downloader.search_datasets("test")
    print(f"Found {len(search_results)} datasets matching 'test':")
    for result in search_results:
        print(f"- {result['name']} ({result['category']})")
    
    print()
    
    # Show final disk usage
    print("💾 Final Disk Usage:")
    final_disk_usage = downloader.get_disk_usage()
    print(f"Total: {final_disk_usage['total']}")
    print(f"Used: {final_disk_usage['used']}")
    print(f"Free: {final_disk_usage['free']}")
    print(f"Usage: {final_disk_usage['usage_percent']:.1f}%")
    
    # Calculate space used by test
    initial_used = float(disk_usage['used'].split()[0]) if disk_usage['used'] else 0
    final_used = float(final_disk_usage['used'].split()[0]) if final_disk_usage['used'] else 0
    space_used = final_used - initial_used
    
    print(f"📊 Space used by test downloads: {space_used:.2f} MB")
    
    print()
    
    # Test cleanup
    print("🧹 Testing cleanup functionality...")
    downloader.cleanup_temp_files()
    print("✅ Cleanup completed!")
    
    # Summary
    print()
    print("🎉 Test Summary:")
    print(f"✅ Successful downloads: {sum([success1, success2, success3])}/3")
    print(f"📁 Datasets stored in: {downloader.base_dir}")
    print(f"📊 Total datasets found: {len(datasets)}")
    
    if sum([success1, success2, success3]) == 3:
        print("🎯 All tests passed! The downloader is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs for details.")
    
    print()
    print("📁 Check your D:/datasets/ directory to see the downloaded files!")
    print("📝 Logs are available in D:/datasets/logs/")

if __name__ == "__main__":
    try:
        test_download()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("Check the logs for more details.") 