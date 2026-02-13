from flask import Flask, request, jsonify
from flask_cors import CORS  
import tensorflow as tf
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)  

try:
    model = tf.keras.models.load_model('2model.h5')
    scaler_data = joblib.load('2scaler_data.sav')
    scaler_target = joblib.load('2scaler_target.sav')
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        features = [
            float(data['gender']), 
            float(data['age']), 
            float(data['tc']), 
            float(data['hdl']), 
            float(data['smoke']), 
            float(data['bpm']), 
            float(data['diab'])
        ]
        
        
        final_features = np.array([features])
        scaled_input = scaler_data.transform(final_features)
        
        prediction = model.predict(scaled_input)

        final_prediction = scaler_target.inverse_transform(prediction)
        
        return jsonify({'score': float(final_prediction[0][0])})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__': 
 app.run(host='0.0.0.0', port=5000)