import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, timedelta
import requests

# Set page config
st.set_page_config(page_title="PRICE_PREDICTOR_AI_MODEL", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .small-font {
        font-size:14px !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Function to get current USD to INR exchange rate
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usd_to_inr_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data['rates']['INR']
    except:
        st.error("Failed to fetch current exchange rate. Using 1 USD = 75 INR as a fallback.")
        return 75  # Fallback exchange rate if API fails

# Function to load CSV data
@st.cache_data  # This decorator will cache the data to improve performance
def load_csv_data(filename):
    return pd.read_csv(filename)

# Function to fetch data from Yahoo Finance
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_yahoo_finance_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Close'].reset_index()

# Dictionary of products, their tickers, and CSV filenames
products = {
    'Wheat': {'ticker': 'ZW=F', 'csv': 'wheat.csv'},
    'Rice': {'ticker': 'ZR=F', 'csv': 'rice.csv'},
    'Corn': {'ticker': 'ZC=F', 'csv': 'corn.csv'},
    'Soybean': {'ticker': 'ZS=F', 'csv': 'soybean.csv'},
    'Cotton': {'ticker': 'CT=F', 'csv': 'cotton.csv'},
    'Sugar': {'ticker': 'SB=F', 'csv': 'sugar.csv'},
    'Coffee': {'ticker': 'KC=F', 'csv': 'coffee.csv'},
    'Cocoa': {'ticker': 'CC=F', 'csv': 'cocoa.csv'},
    'Oats': {'ticker': 'ZO=F', 'csv': 'oats.csv'},
    'Orange Juice': {'ticker': 'OJ=F', 'csv': 'orange_juice.csv'},
}

# Streamlit app
st.markdown('<p class="big-font">Agricultural Product Price Predictor</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<p class="medium-font">Settings</p>', unsafe_allow_html=True)
selected_product = st.sidebar.selectbox('Select a product', list(products.keys()))
data_source = st.sidebar.radio("Choose data source", ("CSV", "Yahoo Finance"))

# Fetch exchange rate
usd_to_inr_rate = get_usd_to_inr_rate()
st.sidebar.markdown(f"<p class='small-font'>Current Exchange Rate: 1 USD = {usd_to_inr_rate:.2f} INR</p>", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f'<p class="medium-font">{selected_product} Price Trends (price per ton (1000kg))</p>', unsafe_allow_html=True)

    # Load data based on selected source
    if data_source == "CSV":
        try:
            df = load_csv_data(products[selected_product]['csv'])
            df.columns = ['Date', 'Price']  # Ensure column names are correct
            st.success(f"CSV data for {selected_product} loaded successfully!")
        except FileNotFoundError:
            st.error(f"CSV file for {selected_product} not found. Please check the file path and name.")
            st.stop()
    else:  # Yahoo Finance
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')  # 6 months of historical data
        df = fetch_yahoo_finance_data(products[selected_product]['ticker'], start_date, end_date)
        df.columns = ['Date', 'Price']  # Rename columns
        st.success(f"Yahoo Finance data for {selected_product} fetched successfully!")

    # Check if data is empty
    if df.empty:
        st.error(f"No data available for {selected_product}.")
    else:
        # Prepare data for model
        df['Date'] = pd.to_datetime(df['Date'])
        df['Price'] = df['Price'] * usd_to_inr_rate  # Convert to INR
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        # Train model
        model = LinearRegression()
        model.fit(df[['Days']], df['Price'])

        # Predict future prices
        future_days = 180  # 6 months
        last_day = df['Days'].max()
        future_dates = pd.date_range(start=df['Date'].max() + timedelta(days=1), periods=future_days)
        future_days_array = np.array(range(last_day + 1, last_day + future_days + 1)).reshape(-1, 1)
        future_prices = model.predict(future_days_array)

        # Combine historical and future data
        all_dates = pd.concat([df['Date'], pd.Series(future_dates)])
        all_prices = pd.concat([df['Price'], pd.Series(future_prices)])
        all_data = pd.DataFrame({'Date': all_dates, 'Price': all_prices})
        all_data['Type'] = ['Historical'] * len(df) + ['Predicted'] * future_days

        # Create Altair chart
        chart = alt.Chart(all_data).mark_line().encode(
            x='Date:T',
            y=alt.Y('Price:Q', scale=alt.Scale(zero=False)),
            color='Type:N'
        ).properties(
            width=700,
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

with col2:
    st.markdown('<p class="medium-font">Price Statistics (price per ton (1000kg))</p>', unsafe_allow_html=True)
    if 'df' in locals() and not df.empty:
        current_price = df['Price'].iloc[-1]
        st.metric("Current Price", f"₹{current_price:.2f}")
        st.metric("Average Price (Last 6 Months)", f"₹{df['Price'].mean():.2f}")
        st.metric("Lowest Price (Last 6 Months)", f"₹{df['Price'].min():.2f}")
        st.metric("Highest Price (Last 6 Months)", f"₹{df['Price'].max():.2f}")

        st.markdown('<p class="medium-font">Price Predictions (price per ton (1000kg))</p>', unsafe_allow_html=True)
        st.metric("Predicted Price (1 month)", f"₹{future_prices[30]:.2f}")
        st.metric("Predicted Price (3 months)", f"₹{future_prices[90]:.2f}")
        st.metric("Predicted Price (6 months)", f"₹{future_prices[-1]:.2f}")

# Explanation and Disclaimer
st.markdown("""
<p class="small-font">
<b>How we calculate prices:</b><br>
1. We use historical commodity prices in USD from either CSV files or Yahoo Finance.<br>
2. These prices are converted to INR using the current exchange rate.<br>
3. Historical data is used to predict future prices using a simple linear regression model.<br><br>
<b>Disclaimer:</b> These predictions are based on historical data and should not be used as the sole basis for financial decisions. 
Many factors can influence agricultural prices. Local market prices may vary due to additional factors not considered in this model.
</p>
""", unsafe_allow_html=True)
