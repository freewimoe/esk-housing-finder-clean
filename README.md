# 🏠 Smart Real Estate Predictor

An intelligent real estate price prediction dashboard using machine learning, geospatial data, and economic indicators.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![Plotly](https://img.shields.io/badge/plotly-v5.0+-green.svg)

## 🎯 Project Overview

### Business Problem
Real estate pricing is complex, influenced by property features, location amenities, economic factors, and market trends. This dashboard provides:
- **Accurate price predictions** using advanced ML models
- **Neighborhood analysis** with walkability and amenity scores
- **Market trend insights** for buyers and investors
- **Interactive exploration** of local real estate markets

### Target Users
- 🏡 **Home Buyers**: Get fair price estimates and neighborhood insights
- 🏢 **Real Estate Agents**: Support client consultations with data-driven insights
- 💰 **Investors**: Identify undervalued properties and growth areas
- 📊 **Market Analysts**: Understand pricing trends and market dynamics

## 🌟 Key Features

### 📍 **Interactive Geospatial Analysis**
- Property locations on interactive maps
- Price heatmaps by neighborhood
- Proximity analysis to amenities (schools, transport, shopping)
- Walkability and safety scores

### 🧠 **Advanced ML Predictions**
- Multi-algorithm ensemble models
- Feature importance analysis
- Confidence intervals for predictions
- "What-if" scenario modeling

### 📈 **Market Intelligence**
- Historical price trends
- Seasonal pattern analysis
- Economic indicator correlations
- Market volatility assessment

### 🏘️ **Neighborhood Profiling**
- School quality ratings
- Crime safety scores
- Transportation accessibility
- Local amenity density

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/freewimoe/streamlit-ml-dashboard.git
cd streamlit-ml-dashboard
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run app/app.py
```

4. **Open your browser**
Navigate to `http://localhost:8501`

## 📊 Sample Data Included

### Try these sample datasets:

**Classification Example (Iris Dataset):**
- Location: `data/raw/sample_iris.csv`
- Target: `species` 
- Features: sepal_length, sepal_width, petal_length, petal_width
- Problem: Multi-class classification

**Regression Example (House Prices):**
- Location: `data/raw/sample_house_prices.csv`
- Target: `price`
- Features: bedrooms, bathrooms, sqft, age
- Problem: Regression

## 📁 Project Structure

```
streamlit-ml-dashboard/
├── app/
│   ├── app.py                 # Main application entry point
│   └── app_pages/
│       ├── __init__.py
│       ├── 1_00_📘_Project_Summary.py
│       ├── 1_01_🔎_EDA.py
│       ├── 1_02_🧠_Train_Model.py
│       ├── 1_03_📈_Predict.py
│       └── 1_04_🧪_Model_Metrics.py
├── data/
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data files
│   └── interim/               # Intermediate data files
├── models/
│   └── versioned/             # Saved model files
├── notebooks/                 # Jupyter notebooks for development
├── src/                       # Source code utilities
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── runtime.txt               # Python runtime version
├── Procfile                  # Deployment configuration
└── README.md                 # This file
```

## 📖 Usage Guide

### 1. **Data Exploration**
- Upload your CSV file in the EDA section
- Explore statistical summaries and visualizations
- Analyze missing values and correlations

### 2. **Model Training**
- Upload training data
- Select target variable and features
- Choose algorithm and parameters
- Train and save your model

### 3. **Making Predictions**
- Use single prediction for individual cases
- Upload CSV for batch predictions
- Download results with confidence scores

### 4. **Model Evaluation**
- Upload test data to evaluate performance
- View comprehensive metrics and visualizations
- Analyze feature importance and model behavior

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/)**: Web app framework
- **[Scikit-learn](https://scikit-learn.org/)**: Machine learning library
- **[Plotly](https://plotly.com/)**: Interactive visualizations
- **[Pandas](https://pandas.pydata.org/)**: Data manipulation
- **[NumPy](https://numpy.org/)**: Numerical computing

## 🚀 Deployment

### Streamlit Cloud
1. Push your code to GitHub
2. Connect your repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Heroku
```bash
# Login to Heroku
heroku login

# Create new app
heroku create your-app-name

# Deploy
git push heroku main
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Streamlit team for the amazing framework
- Scikit-learn contributors for the ML tools
- Plotly for interactive visualizations

---

⭐ **Star this repository if you find it helpful!** ⭐
