from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model/risk_model.pkl")
encoders = joblib.load("model/encoders.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    if request.method == 'POST':

        road_type = request.form['road_type']
        weather = request.form['weather']
        visibility = request.form['visibility']
        traffic_density = request.form['traffic_density']

        temperature = int(request.form['temperature'])
        lanes = int(request.form['lanes'])
        traffic_signal = int(request.form['traffic_signal'])
        is_peak_hour = int(request.form['is_peak_hour'])

        # Encode values
        road_type = encoders["road_type"].transform([road_type])[0]
        weather = encoders["weather"].transform([weather])[0]
        visibility = encoders["visibility"].transform([visibility])[0]
        traffic_density = encoders["traffic_density"].transform([traffic_density])[0]

        data = pd.DataFrame([[
            road_type,
            traffic_signal,
            weather,
            visibility,
            temperature,
            traffic_density,
            lanes,
            is_peak_hour
        ]], columns=[
            "road_type",
            "traffic_signal",
            "weather",
            "visibility",
            "temperature",
            "traffic_density",
            "lanes",
            "is_peak_hour"
        ])

        prediction = model.predict(data)[0]

        if prediction == "High":
            percentage = 85
            recommendation = "High accident risk. Reduce speed and drive carefully."

        elif prediction == "Medium":
            percentage = 60
            recommendation = "Moderate risk. Stay alert and maintain safe distance."

        else:
            percentage = 30
            recommendation = "Low risk. Continue following traffic rules."

        return render_template(
            "result.html",
            prediction=prediction,
            percentage=percentage,
            recommendation=recommendation
        )

    return render_template('predict.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
