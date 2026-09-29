from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

FEATURE_NAMES = [
    'engine_load_pct', 'engine_speed_rpm', 'boost_pressure_bar',
    'air_intake_temp_c', 'air_filter_dp_mbar', 'fuel_rail_pressure_bar',
    'fuel_flow_rate_lph', 'fuel_temp_c', 'lube_oil_pressure_bar',
    'lube_oil_temp_c', 'coolant_temp_c', 'coolant_flow_rate_lpm',
    'exhaust_gas_temp_c', 'crankcase_pressure_mbar', 'vibration_de_mms',
    'vibration_nde_mms'
]

# Helper mock inference in case model binaries are missing during initial deployment
def mock_inference(input_dict):
    load = input_dict.get('engine_load_pct', 75)
    vib = input_dict.get('vibration_de_mms', 1.8)
    temp = input_dict.get('exhaust_gas_temp_c', 450)
    
    # Calculate mock risk based on operational thresholds
    risk_score = min(1.0, max(0.0, (load/100 * 0.3) + (vib/5.0 * 0.4) + (temp/600 * 0.3)))
    fail_48h = 1 if risk_score > 0.75 else 0
    rul_hours = max(10.0, round((1.0 - risk_score) * 500, 1))
    
    return fail_48h, round(risk_score * 100, 2), rul_hours

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json or {}
        input_values = [float(data.get(feat, 0.0)) for feat in FEATURE_NAMES]
        
        # Load models if available
        if (os.path.exists('diesel_scaler.pkl') and 
            os.path.exists('diesel_rf_classifier.pkl') and 
            os.path.exists('diesel_xgb_regressor.pkl')):
            
            scaler = joblib.load('diesel_scaler.pkl')
            rf_classifier = joblib.load('diesel_rf_classifier.pkl')
            xgb_regressor = joblib.load('diesel_xgb_regressor.pkl')
            
            scaled_input = scaler.transform([input_values])
            fail_48h = int(rf_classifier.predict(scaled_input)[0])
            fail_prob = float(rf_classifier.predict_proba(scaled_input)[0][1]) * 100
            rul_hours = float(xgb_regressor.predict(scaled_input)[0])
        else:
            fail_48h, fail_prob, rul_hours = mock_inference(data)
            
        return jsonify({
            'status': 'success',
            'pred_fail_48h': fail_48h,
            'pred_fail_prob': round(fail_prob, 2),
            'pred_rul_hours': round(rul_hours, 2)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
