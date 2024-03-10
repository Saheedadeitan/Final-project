import streamlit as st
import pandas as pd
import numpy as np

pip install pmdarima

import pmdarima as pm

st.title('Auto-ARIMA Forecasting')

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.write("Upload a CSV file.")
    st.stop()

model = pm.auto_arima(data, seasonal=True, m=12)
forecast = model.predict(n_periods=12)  # Forecast the next 12 periods (adjust as needed)

st.line_chart(forecast)
