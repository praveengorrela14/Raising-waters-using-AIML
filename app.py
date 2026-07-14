from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Regional Geographic Database focused exclusively on Andhra Pradesh, India
REGIONAL_DATABASE = {
    "AP_Sector_A": {
        "name": "Godavari Basin (Rajahmundry)", 
        "desc": "High-volume river channel discharge monitoring zone", 
        "lat": 17.0005, "lon": 81.8040
    },
    "AP_Sector_B": {
        "name": "Krishna Delta Area (Vijayawada)", 
        "desc": "Low-lying alluvial agricultural delta floodplain", 
        "lat": 16.5062, "lon": 80.6480
    },
    "AP_Sector_C": {
        "name": "Coastal Vizag Belt (Visakhapatnam)", 
        "desc": "Coastal flash flood and cyclonic storm surge risk area", 
        "lat": 17.6868, "lon": 83.2185
    }
}

try:
    model = joblib.load('best_flood_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    print("CRITICAL ERROR: Pipeline binary files not found. Ensure 'train_model.py' is executed first.")

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = None
    prediction_class = None
    selected_region = "AP_Sector_A"
    map_target = REGIONAL_DATABASE["AP_Sector_A"]
    
    if request.method == 'POST':
        try:
            selected_region = request.form.get('region', 'AP_Sector_A')
            map_target = REGIONAL_DATABASE[selected_region]
            
            # Fetch form parameters
            rainfall = float(request.form['rainfall'])
            visibility = float(request.form['visibility'])
            seasonal = float(request.form['seasonal'])
            
            # Form structured DataFrame matching exact historical weather training column schemas
            input_data = pd.DataFrame([{
                'annual_rainfall': rainfall,
                'cloud_visibility': visibility,
                'seasonal_pattern': seasonal
            }])
            
            # Clean pipeline mapping with zero command console feature warnings
            scaled_features = scaler.transform(input_data)
            prediction = model.predict(scaled_features)
            
            if prediction == 1:
                prediction_text = f"⚠️ CRITICAL FLOOD WARNING: High risk anomaly detected in {map_target['name']}. Deploy regional relief assets."
                prediction_class = "danger"
            else:
                prediction_text = f"✅ NOMINAL STATUS: Conditions remain stable within safe bounds for {map_target['name']}."
                prediction_class = "success"
                
        except Exception as e:
            prediction_text = f"Sensory Intake Error: {str(e)}"
            prediction_class = "error"

    return render_template(
        'index.html', 
        prediction_text=prediction_text, 
        prediction_class=prediction_class,
        selected_region=selected_region,
        map_target=map_target,
        regions=REGIONAL_DATABASE
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
