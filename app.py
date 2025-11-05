import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Solar Predictor",
    page_icon="☀️",
    layout="wide"
)

# --- Model Loading ---
MODEL_FILE = 'solar_model.joblib'
MODEL_FEATURES = ['hour_of_day', 'month', 'temperature_c', 'radiation_w_m2']

@st.cache_resource
def load_model():
    """Loads the trained model. @st.cache_resource ensures this runs only once."""
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        model = joblib.load(MODEL_FILE)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Helper Functions ---
def fetch_weather_data(lat, lon):
    """Fetches live weather data from Open-Meteo."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,shortwave_radiation&hourly=temperature_2m,shortwave_radiation&timezone=auto&forecast_days=1"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None

def get_model_prediction(model, weather_data):
    """
    Runs the model prediction.
    weather_data is a dict with: hour, month, temperature, radiation
    """
    # 1. Convert input data to a DataFrame
    input_data = {
        'hour_of_day': weather_data['hour'],
        'month': weather_data['month'],
        'temperature_c': weather_data['temperature'],
        'radiation_w_m2': weather_data['radiation']
    }
    input_df = pd.DataFrame([input_data])
    
    # 2. Ensure feature order
    input_df = input_df[MODEL_FEATURES]

    # 3. Make prediction
    prediction = model.predict(input_df)
    
    # Ensure prediction is not negative (e.g., at night)
    return max(0, prediction[0])

# --- Main Application UI ---
st.title("☀️ AI Solar Energy Predictor")
st.markdown("This app uses a custom-trained `RandomForestRegressor` model to predict solar energy generation based on live weather data.")
# 

if model is None:
    st.error(f"**Model not found!** Was expecting `{MODEL_FILE}`.")
    st.info("Please run `python create_dataset.py` and `python train_model.py` first to generate and train the model.")
else:
    # --- Sidebar Inputs ---
    st.sidebar.header("Location Input")
    # Use Ikole, Ekiti as the default
    lat = st.sidebar.number_input("Latitude", value=7.79, format="%.2f")
    lon = st.sidebar.number_input("Longitude", value=5.50, format="%.2f")
    st.sidebar.info(f"Default: Ikole, Ekiti, Nigeria. (Current as of {datetime.now().strftime('%b %d, %Y')})")

    if st.sidebar.button("Fetch Weather & Predict", type="primary"):
        with st.spinner("Fetching live weather and running AI model..."):
            
            weather_data = fetch_weather_data(lat, lon)
            
            if weather_data:
                # --- 1. Current Prediction ---
                st.header(f"Current Prediction for ({lat}, {lon})")
                
                current = weather_data['current']
                current_time = datetime.fromisoformat(current['time'])
                
                current_model_input = {
                    'hour': current_time.hour,
                    'month': current_time.month,
                    'temperature': current.get('temperature_2m', 0),
                    'radiation': current.get('shortwave_radiation', 0) or 0
                }
                
                current_prediction_kw = get_model_prediction(model, current_model_input)
                
                st.metric(
                    label=f"Current Generation (as of {current_time.strftime('%I:%M %p')})",
                    value=f"{current_prediction_kw:.2f} kW"
                )

                # --- 2. Hourly Forecast ---
                st.header("24-Hour Generation Forecast")
                
                hourly = weather_data['hourly']
                forecast_data = []
                
                for i in range(len(hourly['time'])):
                    time = datetime.fromisoformat(hourly['time'][i])
                    
                    hourly_model_input = {
                        'hour': time.hour,
                        'month': time.month,
                        'temperature': hourly['temperature_2m'][i],
                        'radiation': hourly.get('shortwave_radiation', [0]*len(hourly['time']))[i] or 0
                    }
                    
                    hourly_prediction_kw = get_model_prediction(model, hourly_model_input)
                    
                    forecast_data.append({
                        'Time': time,
                        'Predicted kW': hourly_prediction_kw
                    })
                
                # Create a DataFrame for charting
                forecast_df = pd.DataFrame(forecast_data)
                forecast_df = forecast_df.set_index('Time')
                
                # Display the chart
                st.line_chart(forecast_df)
                
                # --- 3. Explanation Expander ---
                with st.expander("See Model & Data Details"):
                    st.subheader("Raw Weather Data (Hourly)")
                    st.dataframe(pd.DataFrame(weather_data['hourly']))
                    
                    st.subheader("Model Feature Importance")
                    st.info("This shows which factors the AI model 'learned' were most important.")
                    try:
                        # Display feature importance from the RandomForest model
                        importance = pd.DataFrame({
                            'Feature': MODEL_FEATURES,
                            'Importance': model.feature_importances_
                        }).sort_values(by='Importance', ascending=False)
                        st.bar_chart(importance.set_index('Feature'))
                    except Exception as e:
                        st.write(f"Could not get feature importance: {e}")

    else:
        st.info("Enter coordinates in the sidebar and press 'Predict'.")
