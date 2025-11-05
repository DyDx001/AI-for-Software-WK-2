# ☀️ AI Solar Energy Predictor

**Live Demo:** [intelligent-solar-power-predict.streamlit.app](https://intelligent-solar-power-predict.streamlit.app/)

---

## 🌍 Overview

**AI Solar Energy Predictor** is a full-stack machine learning web application that predicts **solar energy generation (in kW)** for any geographic location.  
It fetches **live weather data** and uses a **trained Scikit-learn model** to produce both real-time predictions and 24-hour forecasts.

This project demonstrates the integration of **AI, data science, and web technologies** for sustainable energy insights.

---

## ⚙️ Key Features

- 🔆 **Live Prediction:** Enter any latitude and longitude to instantly get a real-time energy output estimate.  
- 🕒 **24-Hour Forecast:** Visualizes predicted solar power generation for the next 24 hours using an interactive line chart.  
- 🧠 **Custom AI Model:** Built with a `RandomForestRegressor` trained on five years of synthesized solar-weather data.  
- 🌤️ **Live Data Integration:** Connects to the [Open-Meteo API](https://open-meteo.com/) to fetch real-time temperature and radiation readings.  
- 🧩 **Full Stack ML Pipeline:** From synthetic data generation to model training and deployment on Streamlit Cloud.  

---

## 🧠 Machine Learning Pipeline

This project is divided into three main stages:

### 1. **Data Generation** – `create_dataset.py`
Since public historical solar datasets are limited, this script generates a **realistic 5-year synthesized dataset** (`solar_data.csv`) that models relationships between:
- Hour of the day  
- Month  
- Temperature  
- Radiation  
- Solar power output  

### 2. **Model Training** – `train_model.py`
Trains a **RandomForestRegressor** on the synthetic dataset.  
The model is evaluated using **R² score** and then serialized as `solar_model.joblib` for later inference.

### 3. **Inference Web App** – `app.py`
A **Streamlit** web application that:
1. Loads the trained model (`solar_model.joblib`).  
2. Fetches live weather data via the Open-Meteo API.  
3. Formats data to match model input features.  
4. Runs the model’s prediction pipeline.  
5. Displays:
   - Current solar energy prediction.  
   - 24-hour forecast (interactive chart).  
   - Feature importance and raw weather data.  

---

## 🧰 Tech Stack

- **Python 3.10+**  
- **Streamlit** – Interactive UI and visualization  
- **Scikit-learn** – Machine learning model  
- **Pandas & NumPy** – Data analysis and preprocessing  
- **Joblib** – Model serialization and loading  
- **Requests** – Live weather API integration  

---

## 🚀 Deployment

This app is deployed on **Streamlit Cloud** and accessible here:  
👉 [https://intelligent-solar-power-predict.streamlit.app/](https://intelligent-solar-power-predict.streamlit.app/)

