# AI Dataset Downloader

A comprehensive tool for downloading and generating datasets for AI/ML training, featuring both command-line tools and a user-friendly Streamlit web interface.

## 🚀 Features

### Command Line Tools
- **Universal Dataset Downloader**: Download datasets in multiple formats (CSV, Excel, JSON, ARFF, XML, TSV, TXT)
- **AI Training Dataset Downloader**: Specialized for AI training with popular datasets
- **Large Dataset Downloader**: Download large datasets (5GB+) with progress tracking
- **Batch Downloader**: Download multiple datasets across different categories

### Web Interface (Streamlit)
- **Interactive Dataset Generator**: Generate synthetic datasets on-demand
- **Category Selection**: Choose from 20 different dataset categories
- **Size Control**: Generate datasets from 1MB to 100MB
- **One-Click Download**: Direct CSV download functionality

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-dataset-downloader.git
cd ai-dataset-downloader
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

### Web Interface (Recommended)
```bash
python -m streamlit run dataset_downloader_app.py
```
Then open your browser to `http://localhost:8501`

### Command Line Usage

#### Basic Dataset Download
```bash
python dataset_downloader.py
```

#### AI Training Datasets
```bash
python ai_training_downloader.py
```

#### Large Datasets
```bash
python download_large_csv.py
```

#### Universal Format Support
```bash
python universal_downloader.py
```

## 📊 Available Dataset Categories

### Web Interface Categories
- **Classification**: Binary and multi-class datasets
- **Regression**: Continuous target variables
- **Clustering**: Unsupervised learning datasets
- **NLP**: Text processing and analysis
- **Time Series**: Temporal data analysis
- **Computer Vision**: Image metadata and features
- **Recommendation**: User-item interaction data
- **Anomaly Detection**: Fraud and outlier detection
- **Forecasting**: Predictive modeling data
- **Sentiment Analysis**: Text sentiment classification
- **Fraud Detection**: Financial fraud patterns
- **Customer Behavior**: User interaction analytics
- **Financial**: Market and trading data
- **Healthcare**: Medical and patient data
- **E-commerce**: Sales and product data
- **Social Media**: User engagement metrics
- **Weather**: Climate and meteorological data
- **Traffic**: Transportation and congestion data
- **Energy**: Power consumption patterns
- **Education**: Student performance analytics

### Command Line Datasets
- Iris Dataset (Classification)
- Wine Quality (Regression)
- Diabetes (Regression)
- Breast Cancer (Classification)
- Digits (Classification)
- California Housing (Regression)
- And many more...

## 🛠️ Usage Examples

### Generate a 20MB Classification Dataset
1. Open the Streamlit app
2. Select "classification" from the category dropdown
3. Select "20 MB" from the size dropdown
4. Click "Generate & Download CSV"

### Download Real Datasets
```bash
# Download Iris dataset
python dataset_downloader.py --dataset iris --format csv

# Download multiple datasets
python ai_training_downloader.py --all

# Download large dataset
python download_large_csv.py
```

## 📁 Project Structure

```
ai-dataset-downloader/
├── dataset_downloader_app.py      # Streamlit web interface
├── dataset_downloader.py          # Basic dataset downloader
├── ai_training_downloader.py      # AI training datasets
├── universal_downloader.py        # Multi-format support
├── download_large_csv.py          # Large dataset downloader
├── download_20mb_datasets_fixed.py # Batch dataset generator
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🔧 Configuration

### Custom Dataset URLs
Edit `config.json` to add your own dataset sources:

```json
{
  "datasets": {
    "custom_dataset": {
      "url": "https://example.com/dataset.csv",
      "description": "Custom dataset description",
      "category": "classification"
    }
  }
}
```

### Streamlit Configuration
Create `.streamlit/config.toml` for custom Streamlit settings:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

## 📈 Features in Detail

### Synthetic Dataset Generation
- **Realistic Data**: Generated data follows real-world patterns
- **Configurable Size**: Generate datasets from 1MB to 100MB
- **Multiple Categories**: 20 different dataset types
- **Reproducible**: Fixed random seed for consistent results

### Real Dataset Download
- **Multiple Sources**: UCI, Kaggle, government datasets
- **Format Support**: CSV, Excel, JSON, ARFF, XML, TSV, TXT
- **Progress Tracking**: Real-time download progress
- **Error Handling**: Robust error recovery and retry logic

### Web Interface Features
- **Responsive Design**: Works on desktop and mobile
- **Real-time Generation**: Instant dataset creation
- **Download Integration**: Direct browser download
- **User-friendly**: Intuitive dropdown selections

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for dataset sources
- Streamlit for the web framework
- Pandas and NumPy for data manipulation
- Requests for HTTP operations

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/ai-dataset-downloader/issues) page
2. Create a new issue with detailed information
3. Include your Python version and operating system

## 🔄 Updates

Stay updated with the latest features and improvements by:
- Starring the repository
- Watching for releases
- Following the project

---

**Happy Data Science! 🎉** 