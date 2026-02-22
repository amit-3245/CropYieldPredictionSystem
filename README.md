# 🌾 Crop Yield Prediction System

This is a Machine Learning based Smart Farming System that predicts crop yield using real-time weather data.

##  Features
- Machine Learning Regression Model
- Real-Time Weather API Integration
- Flask Web Application
- Live Crop Yield Prediction

##  Tech Stack
- Python
- Pandas
- Scikit-Learn
- Flask
- OpenWeatherMap API
- HTML, CSS

##  How It Works
User enters:
- State
- Crop
- Area
- City

System fetches:
- Temperature
- Humidity
- Rainfall

ML Model predicts:
- Expected Crop Yield

## ▶ Run Locally
```bash
pip install -r requirements.txt
python train_model.py
python app.py