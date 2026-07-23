# Store Sales Time Series Forecasting

A complete end-to-end time series forecasting project that predicts daily 
grocery sales for Store 1 of Corporación Favorita, a large Ecuadorian 
retailer. Built using Python, Scikit-learn, XGBoost, Prophet, and Streamlit.

---

## Business Context

Accurate sales forecasting helps retail businesses make better inventory 
decisions, reduce waste, and optimize staffing. This project builds a 
forecasting model to predict future grocery sales using historical data, 
oil prices, holiday indicators, and promotional information.

---

## Dataset

The dataset is sourced from the Kaggle Store Sales Time Series Forecasting 
competition. It contains daily sales records for 54 stores across 33 product 
families from January 2013 to August 2017.

Dataset link: https://www.kaggle.com/competitions/store-sales-time-series-forecasting

Note: train.csv is not included in this repository due to GitHub file size 
limits. Download it from Kaggle and place it in the project folder before 
running the notebook or app.

---

## Project Structure
Store_sales_forecasting/
│
├── store_sales_forecasting.ipynb # Main Jupyter Notebook
├── app.py # Streamlit web application
├── xgb_model.pkl # Saved XGBoost model
├── feature_cols.pkl # Saved feature column names
├── model_results.csv # Model comparison results
├── oil.csv # Daily oil prices
├── holidays_events.csv # Ecuadorian holidays
├── stores.csv # Store metadata
├── transactions.csv # Daily transaction counts
└── README.md # Project documentation

---

## Technologies Used

- Python 3.13
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Prophet
- Statsmodels
- pmdarima
- Streamlit
- Joblib

---

## Project Workflow

1. Data Collection and Exploration
2. Data Cleaning and Transformation
3. Exploratory Data Analysis
4. Model Development (SARIMA, Prophet, XGBoost)
5. Hyperparameter Tuning
6. Streamlit Web Application Deployment
7. Documentation and Version Control

---

## Model Performance

| Model | MAE | RMSE | MAPE | Accuracy |
|---|---|---|---|---|
| SARIMA | 463.19 | 620.92 | 20.84% | 79.16% |
| SARIMA Tuned | 570.40 | 703.66 | 27.86% | 72.14% |
| Prophet | 256.50 | 414.55 | 11.16% | 88.84% |
| Prophet Tuned | 297.36 | 433.47 | 14.58% | 85.42% |
| XGBoost | 242.89 | 358.03 | 9.16% | 90.84% |
| XGBoost Tuned | 232.09 | 345.21 | 9.07% | 90.93% |

XGBoost Tuned was selected as the final production model with 90.93% accuracy.

---

## How to Run

### Option 1: Jupyter Notebook
1. Clone the repository
2. Download train.csv from Kaggle and place it in the project folder
3. Install required libraries:
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet statsmodels pmdarima streamlit joblib
4. Open store_sales_forecasting.ipynb and run all cells

### Option 2: Streamlit App
1. Complete the notebook first to generate model files
2. Run the app:
   streamlit run app.py
3. The app will open in your browser at http://localhost:8501

---

## Key Findings

- Strong weekly seasonality confirmed with Wednesday highest and Sunday lowest sales
- Clear upward sales trend from 2014 to 2016 indicating business growth
- April 2016 Manabi earthquake caused the largest sales spike in the dataset
- December is the peak sales month due to Christmas shopping
- XGBoost with engineered lag features and external data outperformed classical time series models
- Hyperparameter tuning improved XGBoost accuracy from 90.84% to 90.93%

---

## Author

Sweta Khadka
Data Science Internship Project