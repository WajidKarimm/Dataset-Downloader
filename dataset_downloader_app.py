import streamlit as st
import pandas as pd
import numpy as np
import io

st.set_page_config(page_title="AI Dataset Downloader", layout="centered")
st.title("📥 AI Dataset Downloader")
st.write("Generate and download synthetic datasets for AI training.")

# Category and size options
categories = [
    "classification", "regression", "clustering", "nlp", "time_series", "computer_vision",
    "recommendation", "anomaly_detection", "forecasting", "sentiment_analysis", "fraud_detection",
    "customer_behavior", "financial", "healthcare", "ecommerce", "social_media",
    "weather", "traffic", "energy", "education"
]

size_options = {
    "1 MB": 1,
    "5 MB": 5,
    "20 MB": 20,
    "100 MB": 100
}

category = st.selectbox("Select dataset category", categories)
size_label = st.selectbox("Select dataset size", list(size_options.keys()))
size_mb = size_options[size_label]

st.write(f"**Category:** {category}")
st.write(f"**Target size:** {size_mb} MB")

def generate_dataset(category, size_mb):
    target_bytes = size_mb * 1024 * 1024
    estimated_rows = min(target_bytes // 200, 100000)
    np.random.seed(42)
    def nrows(cap):
        return int(min(estimated_rows, cap))
    if category == "classification":
        n = estimated_rows
        features = np.random.randn(n, 20)
        labels = np.random.choice([0, 1], n, p=[0.7, 0.3])
        df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = labels
    elif category == "regression":
        n = estimated_rows
        features = np.random.randn(n, 15)
        target = features[:, 0] * 2 + features[:, 1] * 1.5 + np.random.normal(0, 0.1, n)
        df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(15)])
        df['target'] = target
    elif category == "clustering":
        n = estimated_rows
        n1 = n // 3
        n2 = n // 3
        n3 = n - n1 - n2
        cluster1 = np.random.normal([0, 0], [1, 1], (n1, 2))
        cluster2 = np.random.normal([5, 5], [1, 1], (n2, 2))
        cluster3 = np.random.normal([2, 8], [1, 1], (n3, 2))
        features = np.vstack([cluster1, cluster2, cluster3])
        df = pd.DataFrame(features, columns=['feature_1', 'feature_2'])
    elif category == "nlp":
        n = estimated_rows
        words = ['data', 'science', 'machine', 'learning', 'artificial', 'intelligence', 'algorithm', 'model', 'prediction', 'analysis']
        texts = []
        for _ in range(n):
            text = ' '.join(np.random.choice(words, np.random.randint(5, 15)))
            texts.append(text)
        df = pd.DataFrame({'text': texts, 'length': [len(t) for t in texts]})
    elif category == "time_series":
        n = nrows(10000)
        dates = pd.date_range('2020-01-01', periods=n, freq='H')
        values = np.cumsum(np.random.randn(n)) + 100
        df = pd.DataFrame({'timestamp': dates, 'value': values})
    elif category == "financial":
        n = nrows(10000)
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        prices = 100 + np.cumsum(np.random.randn(n) * 0.02)
        volumes = np.random.randint(1000, 10000, n)
        df = pd.DataFrame({
            'date': dates, 
            'price': prices, 
            'volume': volumes,
            'returns': np.diff(prices, prepend=prices[0])
        })
    elif category == "healthcare":
        n = estimated_rows
        ages = np.random.randint(18, 80, n)
        bmi = np.random.normal(25, 5, n)
        blood_pressure = np.random.normal(120, 20, n)
        cholesterol = np.random.normal(200, 40, n)
        df = pd.DataFrame({
            'age': ages,
            'bmi': bmi,
            'blood_pressure': blood_pressure,
            'cholesterol': cholesterol,
            'risk_score': (ages * 0.1 + (bmi - 25) * 0.2 + (blood_pressure - 120) * 0.01 + (cholesterol - 200) * 0.005)
        })
    elif category == "ecommerce":
        n = estimated_rows
        product_ids = np.random.randint(1, 1000, n)
        prices = np.random.uniform(10, 500, n)
        quantities = np.random.randint(1, 10, n)
        customer_ids = np.random.randint(1, 10000, n)
        df = pd.DataFrame({
            'product_id': product_ids,
            'price': prices,
            'quantity': quantities,
            'customer_id': customer_ids,
            'total_amount': prices * quantities
        })
    elif category == "social_media":
        n = estimated_rows
        user_ids = np.random.randint(1, 50000, n)
        post_lengths = np.random.randint(10, 500, n)
        likes = np.random.poisson(50, n)
        shares = np.random.poisson(10, n)
        df = pd.DataFrame({
            'user_id': user_ids,
            'post_length': post_lengths,
            'likes': likes,
            'shares': shares,
            'engagement_rate': (likes + shares) / post_lengths
        })
    elif category == "weather":
        n = nrows(10000)
        dates = pd.date_range('2020-01-01', periods=n, freq='H')
        temperatures = np.random.normal(20, 10, n)
        humidity = np.random.uniform(30, 90, n)
        pressure = np.random.normal(1013, 20, n)
        df = pd.DataFrame({
            'timestamp': dates,
            'temperature': temperatures,
            'humidity': humidity,
            'pressure': pressure
        })
    elif category == "traffic":
        n = nrows(10000)
        timestamps = pd.date_range('2020-01-01', periods=n, freq='15min')
        vehicles = np.random.poisson(100, n)
        speed = np.random.normal(60, 15, n)
        congestion = np.random.uniform(0, 1, n)
        df = pd.DataFrame({
            'timestamp': timestamps,
            'vehicles_per_hour': vehicles,
            'average_speed': speed,
            'congestion_level': congestion
        })
    elif category == "energy":
        n = nrows(10000)
        timestamps = pd.date_range('2020-01-01', periods=n, freq='H')
        consumption = np.random.normal(1000, 200, n) + np.sin(np.arange(n) * 2 * np.pi / 24) * 200
        solar_generation = np.maximum(0, np.random.normal(500, 100, n) * np.sin(np.arange(n) * 2 * np.pi / 24))
        df = pd.DataFrame({
            'timestamp': timestamps,
            'consumption_kwh': consumption,
            'solar_generation_kwh': solar_generation,
            'net_consumption': consumption - solar_generation
        })
    elif category == "education":
        n = estimated_rows
        student_ids = np.random.randint(1, 10000, n)
        study_hours = np.random.normal(5, 2, n)
        attendance = np.random.uniform(0.7, 1.0, n)
        gpa = np.random.normal(3.0, 0.5, n)
        df = pd.DataFrame({
            'student_id': student_ids,
            'study_hours_per_day': study_hours,
            'attendance_rate': attendance,
            'gpa': gpa,
            'performance_score': gpa * attendance * (study_hours / 5)
        })
    elif category == "computer_vision":
        n = estimated_rows
        image_ids = np.random.randint(1, 100000, n)
        widths = np.random.randint(100, 2000, n)
        heights = np.random.randint(100, 2000, n)
        brightness = np.random.uniform(0.3, 1.0, n)
        contrast = np.random.uniform(0.5, 1.5, n)
        df = pd.DataFrame({
            'image_id': image_ids,
            'width': widths,
            'height': heights,
            'brightness': brightness,
            'contrast': contrast,
            'aspect_ratio': widths / heights
        })
    elif category == "recommendation":
        n = nrows(10000)
        user_ids = np.random.randint(1, 10000, n)
        item_ids = np.random.randint(1, 5000, n)
        ratings = np.random.randint(1, 6, n)
        timestamps = pd.date_range('2020-01-01', periods=n, freq='min')
        df = pd.DataFrame({
            'user_id': user_ids,
            'item_id': item_ids,
            'rating': ratings,
            'timestamp': timestamps
        })
    elif category == "anomaly_detection":
        n = estimated_rows
        normal_data = np.random.normal(0, 1, n - 100)
        anomalies = np.random.normal(5, 1, 100)
        values = np.concatenate([normal_data, anomalies])
        np.random.shuffle(values)
        df = pd.DataFrame({
            'value': values,
            'is_anomaly': [1 if v > 3 else 0 for v in values]
        })
    elif category == "forecasting":
        n = nrows(10000)
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        trend = np.arange(n) * 0.1
        seasonality = np.sin(np.arange(n) * 2 * np.pi / 365) * 10
        noise = np.random.normal(0, 2, n)
        values = 100 + trend + seasonality + noise
        df = pd.DataFrame({
            'date': dates,
            'value': values,
            'trend': trend,
            'seasonality': seasonality
        })
    elif category == "sentiment_analysis":
        n = estimated_rows
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'worst']
        neutral_words = ['okay', 'fine', 'average', 'normal', 'standard', 'regular']
        sentiments = []
        texts = []
        for _ in range(n):
            if np.random.random() < 0.4:
                words = np.random.choice(positive_words, np.random.randint(3, 8))
                sentiment = 1
            elif np.random.random() < 0.7:
                words = np.random.choice(negative_words, np.random.randint(3, 8))
                sentiment = 0
            else:
                words = np.random.choice(neutral_words, np.random.randint(3, 8))
                sentiment = 0.5
            texts.append(' '.join(words))
            sentiments.append(sentiment)
        df = pd.DataFrame({
            'text': texts,
            'sentiment': sentiments
        })
    elif category == "fraud_detection":
        n = nrows(10000)
        amounts = np.random.exponential(100, n)
        locations = np.random.randint(1, 100, n)
        times = pd.date_range('2020-01-01', periods=n, freq='min')
        is_fraud = np.random.choice([0, 1], n, p=[0.95, 0.05])
        amounts[is_fraud == 1] *= np.random.uniform(2, 5, sum(is_fraud))
        df = pd.DataFrame({
            'amount': amounts,
            'location_id': locations,
            'timestamp': times,
            'is_fraud': is_fraud
        })
    elif category == "customer_behavior":
        n = estimated_rows
        customer_ids = np.random.randint(1, 50000, n)
        session_duration = np.random.exponential(300, n)
        pages_visited = np.random.poisson(5, n)
        purchase_amount = np.random.exponential(50, n)
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'session_duration_seconds': session_duration,
            'pages_visited': pages_visited,
            'purchase_amount': purchase_amount,
            'conversion_rate': (purchase_amount > 0).astype(int)
        })
    else:
        n = estimated_rows
        features = np.random.randn(n, 20)
        df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(20)])
    return df

if st.button("Generate & Download CSV"):
    with st.spinner("Generating dataset..."):
        df = generate_dataset(category, size_mb)
        csv = df.to_csv(index=False).encode('utf-8')
        st.success(f"Dataset ready! Shape: {df.shape}")
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"{category}_{size_mb}MB.csv",
            mime="text/csv"
        ) 