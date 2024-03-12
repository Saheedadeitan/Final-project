import streamlit as st
import joblib
import pmdarima as pm

uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.write("Upload a CSV file.")
    st.stop()


model = pm.auto_arima(data.Sales, start_p=0, start_q=0, max_p=3, max_q=3, m=12,max_P=2,d=1,
    max_D=1,max_Q=2,start_P=0, seasonal=True, trace=True,
                      error_action='ignore', 
                      suppress_warnings=True, 
                      stepwise=True) 
st.title('Auto-ARIMA Forecasting')

if len(forecast) > 0:
    st.write(f"Estimated Price: ${forecast.iloc[0]:,.2f}")
else:
    st.write("No forecast available.")

forecast = model.predict(n_periods=6)
st.line_chart(forecast)

  # Display the prediction
st.write(f"Estimated Price: ${forecast.iloc[0]:,.2f}")

print(forecast)
