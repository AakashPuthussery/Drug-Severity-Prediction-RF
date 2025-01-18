from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
model = joblib.load('addiction_risk_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    return 'Welcome to the Addiction Risk Prediction API!'

@app.route('/predict', methods=['POST'])
def predict_risk():
    data = request.get_json()

    # Preprocess the input data
    new_data = pd.DataFrame([data])
    new_data['Gender'] = label_encoder.transform(new_data['Gender'])
    new_data_scaled = scaler.transform(new_data)

    # Make prediction
    prediction = model.predict(new_data_scaled)

    # Return the result as a JSON response
    return jsonify({'Addiction_Risk': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
