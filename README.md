# Diesel Engine Predictive Maintenance Dashboard

A Streamlit web application for monitoring operational metrics and forecasting diesel engine failure risks within a 48-hour window. The dashboard provides real-time risk scoring, parameter sensitivity simulations, performance analytics, and report exports.

## Features

- **Failure Forecasting**: Calculates real-time failure probabilities based on 16 engine telemetry parameters (vibration, thermal gradients, oil pressure, and fuel dynamics).
- **Preset Scenarios**: Quickly load nominal, average, or high-risk operational profiles to test dashboard response.
- **Sensitivity Analysis**: Interactive what-if controls to sweep individual parameters and identify critical failure thresholds.
- **Model Performance Metrics**: Built-in diagnostics including confusion matrix displays and feature importance rankings.
- **Audit Logging & Export**: Logs predictions during the session and exports operational reports to CSV.

## Repository Structure

```text
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── model.pkl           # Trained ML model (optional)
├── scaler.pkl          # Feature scaler (optional)
└── README.md
