# F1-Championships-Analysis

![Formula 1](https://img.shields.io/badge/Formula%201-Analysis-red)
![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Machine Learning](https://img.shields.io/badge/ML-Sklearn-orange)
![Data Science](https://img.shields.io/badge/Data%20Science-Pandas-green)

## Overview

Comprehensive Exploratory Data Analysis (EDA) and Machine Learning analysis of Formula 1 Championships spanning from 1950 to 2025. This project explores 75+ years of F1 racing history, uncovering patterns, trends, and building predictive models for championship outcomes and driver performance.

## Dataset

The project uses the **Formula 1 Championships (1950-2025)** dataset from Kaggle:
- **Source**: [rockyt07/formula-1-championships-1950-2025](https://www.kaggle.com/datasets/rockyt07/formula-1-championships-1950-2025)
- **Time Period**: 1950 - 2025
- **Content**: Race results, driver statistics, constructor data, circuit information, and championship standings

## Project Structure

```
F1-Championships-Analysis/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── download_dataset.py          # Dataset download script
├── F1_Analysis.ipynb           # Main Jupyter notebook with EDA & ML
│
└── data/                        # Dataset files (auto-downloaded)
```

## Features

### 1. Data Loading & Preprocessing
- Automated dataset download using `kagglehub`
- Data cleaning and transformation
- Handling missing values and outliers
- Feature engineering for ML models

### 2. Exploratory Data Analysis (EDA)
- Championship trends over 75 years
- Driver performance analysis
- Constructor dominance patterns
- Circuit statistics and race outcomes
- Historical win rate analysis
- Podium finish distributions

### 3. Statistical Analysis
- Correlation analysis between features
- Time-series analysis of championship evolution
- Statistical significance testing
- Performance metrics comparison

### 4. Machine Learning Models
- **Classification Models**:
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Logistic Regression
- **Prediction Tasks**:
  - Race winner prediction
  - Championship outcome forecasting
  - Driver performance classification

### 5. Visualizations
- Interactive plots using Matplotlib and Seaborn
- Championship evolution timelines
- Performance heatmaps
- Trend analysis charts

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Kaggle API credentials (for dataset download)

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Dheerajjanaswamy/F1-Championships-Analysis.git
   cd F1-Championships-Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Kaggle API** (if not already done):
   - Get your API key from [Kaggle Account Settings](https://www.kaggle.com/settings)
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Username>\.kaggle\` (Windows)

## Usage

### Method 1: Download Dataset First

```bash
python download_dataset.py
```

This will:
- Download the F1 Championships dataset from Kaggle
- Display the dataset path
- List all downloaded files

### Method 2: Use Jupyter Notebook Directly

Open and run the Jupyter notebook:

```bash
jupyter notebook F1_Analysis.ipynb
```

The notebook includes:
1. Automated dataset download
2. Complete EDA workflow
3. Machine learning model training
4. Results visualization

## Key Insights

The analysis reveals:
- Evolution of F1 racing over 75 years
- Dominant eras and legendary drivers
- Constructor performance trends
- Circuit characteristics and their impact
- Predictive patterns for championship outcomes

## Technologies Used

- **Python 3.9+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter**: Interactive notebook environment
- **KaggleHub**: Dataset downloading

## Machine Learning Results

The project implements multiple ML algorithms to predict:
- Race outcomes with accuracy metrics
- Championship winner predictions
- Driver performance classifications

Results include:
- Model accuracy scores
- Confusion matrices
- Feature importance analysis
- Cross-validation results

## Contributing

Contributions are welcome! Please feel free to:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Future Enhancements

- [ ] Add deep learning models for race prediction
- [ ] Include weather data analysis
- [ ] Implement real-time race prediction dashboard
- [ ] Add more advanced feature engineering
- [ ] Create interactive web visualizations

## License

This project is open source and available for educational purposes.

## Acknowledgments

- Dataset provided by [rockyt07](https://www.kaggle.com/rockyt07) on Kaggle
- Formula 1 historical data community
- Open source ML and data science community

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Happy Racing! 🏎️🏁**
