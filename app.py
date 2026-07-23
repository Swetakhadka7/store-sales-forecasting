import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title='Store Sales Forecasting',
    page_icon='📈',
    layout='wide'
)

# Load the saved model and feature columns
model = joblib.load('xgb_model.pkl')
feature_cols = joblib.load('feature_cols.pkl')
model_results = pd.read_csv('model_results.csv')

# App title
st.title('Store Sales Time Series Forecasting')
st.write('Interactive dashboard for grocery sales forecasting - Store 1, GROCERY I')

# Sidebar
st.sidebar.header('Forecast Settings')

forecast_days = st.sidebar.slider(
    'Select Forecast Horizon (days)',
    min_value=7,
    max_value=90,
    value=30,
    step=7
)

st.sidebar.write('---')
st.sidebar.subheader('Historical Data Range')

start_year = st.sidebar.selectbox('Start Year',
    options=[2013, 2014, 2015, 2016, 2017], index=0)

end_year = st.sidebar.selectbox('End Year',
    options=[2013, 2014, 2015, 2016, 2017], index=4)

st.sidebar.write('---')
st.sidebar.subheader('Model Performance')
st.sidebar.dataframe(model_results[['Model', 'Accuracy']].set_index('Model'))

# Load and prepare historical data
@st.cache_data
def load_data():
    train = pd.read_csv('train.csv', parse_dates=['date'])
    df = train[(train['store_nbr'] == 1) & 
               (train['family'] == 'GROCERY I')].copy()
    df = df.set_index('date')
    df = df[['sales', 'onpromotion']]
    full_date_range = pd.date_range(start=df.index.min(),
                                     end=df.index.max(), freq='D')
    df = df.reindex(full_date_range)
    df['sales'] = df['sales'].replace(0, np.nan).interpolate().bfill()
    df['onpromotion'] = df['onpromotion'].fillna(0)
    return df

df_hist = load_data()

# Historical sales chart with year filter
st.subheader('Historical Sales Data')

filtered_hist = df_hist['sales'][
    (df_hist.index.year >= start_year) &
    (df_hist.index.year <= end_year)
]

st.line_chart(filtered_hist)

# Generate forecast
def generate_forecast(df, days):
    last_date = df.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                  periods=days, freq='D')
    future_df = pd.DataFrame(index=future_dates)
    future_df['onpromotion'] = 0
    future_df['dcoilwtico'] = df_hist['onpromotion'].mean()
    future_df['is_holiday'] = 0
    future_df['transactions'] = df_hist['sales'].mean()
    future_df['day_of_week'] = future_df.index.dayofweek
    future_df['day_of_month'] = future_df.index.day
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['week_of_year'] = future_df.index.isocalendar().week.astype(int)
    future_df['quarter'] = future_df.index.quarter
    future_df['is_weekend'] = (future_df.index.dayofweek >= 5).astype(int)

    recent_sales = df['sales'].tail(365).values

    for i in range(days):
        if i == 0:
            future_df.loc[future_dates[i], 'lag_1'] = recent_sales[-1]
            future_df.loc[future_dates[i], 'lag_7'] = recent_sales[-7]
            future_df.loc[future_dates[i], 'lag_14'] = recent_sales[-14]
            future_df.loc[future_dates[i], 'lag_30'] = recent_sales[-30]
            future_df.loc[future_dates[i], 'lag_365'] = recent_sales[-365]
            future_df.loc[future_dates[i], 'rolling_mean_7'] = np.mean(recent_sales[-7:])
            future_df.loc[future_dates[i], 'rolling_mean_30'] = np.mean(recent_sales[-30:])
            future_df.loc[future_dates[i], 'rolling_std_7'] = np.std(recent_sales[-7:])
        else:
            all_sales = np.append(recent_sales,
                                  future_df['lag_1'].iloc[:i].values)
            future_df.loc[future_dates[i], 'lag_1'] = all_sales[-1]
            future_df.loc[future_dates[i], 'lag_7'] = all_sales[-7] if len(all_sales) >= 7 else recent_sales[-7]
            future_df.loc[future_dates[i], 'lag_14'] = all_sales[-14] if len(all_sales) >= 14 else recent_sales[-14]
            future_df.loc[future_dates[i], 'lag_30'] = all_sales[-30] if len(all_sales) >= 30 else recent_sales[-30]
            future_df.loc[future_dates[i], 'lag_365'] = recent_sales[-365+i] if len(recent_sales) >= 365 else recent_sales[-1]
            future_df.loc[future_dates[i], 'rolling_mean_7'] = np.mean(all_sales[-7:])
            future_df.loc[future_dates[i], 'rolling_mean_30'] = np.mean(all_sales[-30:])
            future_df.loc[future_dates[i], 'rolling_std_7'] = np.std(all_sales[-7:])

    predictions = model.predict(future_df[feature_cols])
    return future_dates, predictions

future_dates, predictions = generate_forecast(df_hist, forecast_days)

# Display forecast chart
st.subheader(f'Sales Forecast for Next {forecast_days} Days')

fig, ax = plt.subplots(figsize=(15, 6))

historical_tail = df_hist['sales'].tail(90)
ax.plot(historical_tail.index, historical_tail.values,
        label='Historical Sales', color='steelblue', linewidth=1.5)

ax.plot(future_dates, predictions,
        label='Forecasted Sales', color='red',
        linewidth=1.5, linestyle='--')

ax.fill_between(future_dates,
                predictions * 0.9,
                predictions * 1.1,
                alpha=0.2, color='red',
                label='Uncertainty Interval (10%)')

ax.axvline(x=df_hist.index.max(), color='gray',
           linestyle=':', linewidth=1, label='Forecast Start')
ax.set_title('Historical Sales and Future Forecast')
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.legend()
plt.tight_layout()
st.pyplot(fig)

# Forecast metrics
st.subheader('Forecast Summary')
col1, col2, col3 = st.columns(3)

with col1:
    st.metric('Average Forecasted Sales', f'${np.mean(predictions):,.0f}')

with col2:
    st.metric('Peak Forecasted Sales', f'${np.max(predictions):,.0f}')

with col3:
    st.metric('Minimum Forecasted Sales', f'${np.min(predictions):,.0f}')

# Download forecast results
st.subheader('Download Forecast Results')
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted Sales': predictions.round(2)
})

csv = forecast_df.to_csv(index=False)
st.download_button(
    label='Download Forecast as CSV',
    data=csv,
    file_name='sales_forecast.csv',
    mime='text/csv'
)

# Model performance section
st.subheader('Model Performance Comparison')
st.dataframe(model_results.set_index('Model'))