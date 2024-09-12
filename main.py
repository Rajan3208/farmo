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

# Custom CSS (unchanged)
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

# Function to fetch historical data
def fetch_historical_data(product_ticker, start_date, end_date):
    data = yf.download(product_ticker, start=start_date, end=end_date)
    return data['Close']

# Function to get current USD to INR exchange rate
def get_usd_to_inr_rate():
    try:
        response = requests.get("https://api.exchangerate-api.com/v4/latest/USD")
        data = response.json()
        return data['rates']['INR']
    except:
        st.error("Failed to fetch current exchange rate. Using 1 USD = 75 INR as a fallback.")
        return 75  # Fallback exchange rate if API fails

# Dictionary of products, their tickers
products = {
    'Wheat': 'ZW=F',
    'Rice': 'RR=F',
    'Corn': 'ZC=F',
    'Soybean': 'ZS=F',
    'Cotton': 'CT=F',
    'Mustard': 'RS=F',
    'Sugar': 'SB=F',
    'Coffee': 'KC=F',
    'Cocoa': 'CC=F',
    'Barley': 'K1=F',
    'Oats': 'ZO=F',
    'Canola': 'RS=F',
    'Pork': 'HE=F',
    'Orange Juice': 'OJ=F',
    'Peanuts': 'N=F',
    'Potatoes': 'POTATO=F',
}

# Streamlit app
st.markdown('<p class="big-font">Agricultural Product Price Predictor</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown('<p class="medium-font">Settings</p>', unsafe_allow_html=True)
selected_product = st.sidebar.selectbox('Select a product', list(products.keys()))

# Fetch exchange rate
usd_to_inr_rate = get_usd_to_inr_rate()
st.sidebar.markdown(f"<p class='small-font'>Current Exchange Rate: 1 USD = {usd_to_inr_rate:.2f} INR</p>", unsafe_allow_html=True)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f'<p class="medium-font">{selected_product} Price Trends (price per ton (1000kg))</p>', unsafe_allow_html=True)

    # Fetch data for the last 6 months
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')  # 6 months of historical data

    # Fetch historical data
    data = fetch_historical_data(products[selected_product], start_date, end_date)

    # Check if data is empty
    if data.empty:
        st.error(f"No historical data available for {selected_product}.")
    else:
        # Convert prices to INR
        data_inr = data * usd_to_inr_rate  # Convert to INR

        # Prepare data for model
        df = pd.DataFrame(data_inr)
        df.reset_index(inplace=True)
        df.columns = ['Date', 'Price']

        # Convert 'Date' to datetime format before calculating 'Days'
        df['Date'] = pd.to_datetime(df['Date'])

        # Calculate the number of days
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days

        # Check if there is enough data to train the model
        if df.empty or len(df) < 2:
            st.error("Not enough data to train the model.")
        else:
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
1. We fetch international commodity prices in USD.<br>
2. These prices are converted to INR using the current exchange rate.<br>
3. Historical data is used to predict future prices using a simple linear regression model.<br><br>
<b>Disclaimer:</b> These predictions are based on historical data and should not be used as the sole basis for financial decisions. 
Many factors can influence agricultural prices. Local market prices may vary due to additional factors not considered in this model.
</p>
""", unsafe_allow_html=True)