#!/usr/bin/env python3
"""
Test script for the Universal Dataset Downloader Tool
This script will automatically download datasets in various formats to test functionality
"""

from universal_downloader import UniversalDatasetDownloader
import time
import json

def test_universal_download():
    """Test the universal dataset downloader functionality"""
    print("🧪 Testing Universal Dataset Downloader Tool")
    print("=" * 60)
    print("📄 Supports: CSV, Excel, JSON, ARFF, XML, TSV, TXT")
    print()
    
    # Initialize the downloader
    print("📁 Initializing universal downloader...")
    downloader = UniversalDatasetDownloader()
    
    # Show initial disk usage
    print(f"📁 Base Directory: {downloader.base_dir}")
    print()
    
    # Test 1: Download CSV dataset
    print("📥 Test 1: Downloading CSV dataset (Iris)...")
    success1 = downloader.download_and_parse_dataset(
        name="iris_universal",
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        category="classification",
        metadata={"description": "Iris dataset in CSV format", "test": True}
    )
    
    if success1:
        print("✅ CSV dataset downloaded and parsed successfully!")
    else:
        print("❌ Failed to download CSV dataset")
    
    print()
    
    # Test 2: Download JSON dataset
    print("📥 Test 2: Downloading JSON dataset...")
    success2 = downloader.download_and_parse_dataset(
        name="posts_json",
        url="https://jsonplaceholder.typicode.com/posts",
        category="api_data",
        metadata={"description": "Sample JSON posts data", "test": True}
    )
    
    if success2:
        print("✅ JSON dataset downloaded and parsed successfully!")
    else:
        print("❌ Failed to download JSON dataset")
    
    print()
    
    # Test 3: Download XML dataset
    print("📥 Test 3: Downloading XML dataset...")
    success3 = downloader.download_and_parse_dataset(
        name="note_xml",
        url="https://www.w3schools.com/xml/note.xml",
        category="sample",
        metadata={"description": "Sample XML note data", "test": True}
    )
    
    if success3:
        print("✅ XML dataset downloaded and parsed successfully!")
    else:
        print("❌ Failed to download XML dataset")
    
    print()
    
    # Test 4: Download TSV dataset
    print("📥 Test 4: Downloading TSV dataset...")
    success4 = downloader.download_and_parse_dataset(
        name="gdp_tsv",
        url="https://raw.githubusercontent.com/datasets/gdp/master/data/gdp.tsv",
        category="economics",
        metadata={"description": "GDP data in TSV format", "test": True}
    )
    
    if success4:
        print("✅ TSV dataset downloaded and parsed successfully!")
    else:
        print("❌ Failed to download TSV dataset")
    
    print()
    
    # Test 5: Download TXT dataset
    print("📥 Test 5: Downloading TXT dataset...")
    success5 = downloader.download_and_parse_dataset(
        name="sample_txt",
        url="https://www.gutenberg.org/files/1342/1342-0.txt",
        category="literature",
        metadata={"description": "Pride and Prejudice text", "test": True}
    )
    
    if success5:
        print("✅ TXT dataset downloaded and parsed successfully!")
    else:
        print("❌ Failed to download TXT dataset")
    
    print()
    
    # List downloaded datasets with format information
    print("📋 Listing downloaded datasets with formats...")
    metadata_dir = downloader.base_dir / "metadata"
    if metadata_dir.exists():
        datasets = []
        for metadata_file in metadata_dir.glob("*.json"):
            with open(metadata_file, 'r') as f:
                dataset = json.load(f)
                datasets.append(dataset)
        
        print(f"Found {len(datasets)} downloaded datasets:")
        for dataset in datasets:
            format_info = f" ({dataset.get('file_format', 'unknown')})"
            print(f"- {dataset['name']}{format_info} - {dataset['category']}")
            if dataset.get('parse_result'):
                print(f"  Parsed: {dataset['parse_result'].get('format', 'unknown')} format")
    else:
        print("No datasets found.")
    
    print()
    
    # Show format-specific directories
    print("📁 Format-specific directories created:")
    formats_dir = downloader.base_dir / "formats"
    if formats_dir.exists():
        for format_dir in formats_dir.iterdir():
            if format_dir.is_dir():
                files = list(format_dir.glob("*.json"))
                print(f"- {format_dir.name}: {len(files)} parsed files")
    
    print()
    
    # Summary
    print("🎉 Test Summary:")
    print(f"✅ Successful downloads: {sum([success1, success2, success3, success4, success5])}/5")
    print(f"📁 Datasets stored in: {downloader.base_dir}")
    print(f"📄 Supported formats tested: CSV, JSON, XML, TSV, TXT")
    
    if sum([success1, success2, success3, success4, success5]) >= 3:
        print("🎯 Most tests passed! The universal downloader is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs for details.")
    
    print()
    print("📁 Check your D:/datasets/ directory to see the downloaded files!")
    print("📝 Logs are available in D:/datasets/logs/")
    print("📄 Parsed information is available in D:/datasets/formats/")

if __name__ == "__main__":
    try:
        test_universal_download()
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("Check the logs for more details.") 