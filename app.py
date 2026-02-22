from flask import Flask, render_template, request
import joblib
import requests
import numpy as np

app = Flask(__name__)

model = joblib.load('model/crop_yield_model.pkl')
state_encoder = joblib.load('model/state_encoder.pkl')
crop_encoder = joblib.load('model/crop_encoder.pkl')

#  Add your OpenWeatherMap API Key here
API_KEY = "50aad461db824eb2cba747c9c9f3b101"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    state = request.form['state']
    crop = request.form['crop']
    area = float(request.form['area'])
    city = request.form['city']

    # 🌦 Fetch Live Weather Data
    weather_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(weather_url)
    weather_data = response.json()

    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']

    try:
        rainfall = weather_data['rain']['1h']
    except:
        rainfall = 0

    # Encode Inputs
    state = state_encoder.transform([state])[0]
    crop = crop_encoder.transform([crop])[0]

    # Feature Array
    features = np.array([[state, crop, area, temp, humidity, rainfall]])

    prediction = model.predict(features)

    return render_template('index.html',
        prediction_text=f"Predicted Yield: {prediction[0]:.2f} Tons")

if __name__ == "__main__":
    app.run(debug=True)