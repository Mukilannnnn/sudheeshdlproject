import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Correct file paths
model = load_model('music_emotion_model.h5')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the selected important features
features = [
    "_HarmonicChangeDetectionFunction_Std",
    "_Zero-crossingrate_Mean",
    "_HarmonicChangeDetectionFunction_PeriodAmp",
    "_Fluctuation_Mean",
    "_HarmonicChangeDetectionFunction_Mean",
    "_EntropyofSpectrum_Mean",
    "_Pulseclarity_Mean",
    "_Eventdensity_Mean"
]

# Streamlit app
st.title('Music Emotion Classification')

# Attempt to load feature ranges from CSV file
try:
    df = pd.read_csv(r'Acoustic Features.csv')
    feature_ranges = {
        feature: {'min': df[feature].min(), 'max': df[feature].max()} for feature in features
    }
except FileNotFoundError:
    st.error('The Acoustic Features.csv file was not found. Using predefined feature ranges.')
    # Predefined feature ranges as a fallback
    feature_ranges = {
        "_HarmonicChangeDetectionFunction_Std": {'min': 0.01, 'max': 0.15},
        "_Zero-crossingrate_Mean": {'min': 0.02, 'max': 0.20},
        "_HarmonicChangeDetectionFunction_PeriodAmp": {'min': 0.01, 'max': 0.25},
        "_Fluctuation_Mean": {'min': 0.01, 'max': 0.10},
        "_HarmonicChangeDetectionFunction_Mean": {'min': 0.01, 'max': 0.18},
        "_EntropyofSpectrum_Mean": {'min': 0.05, 'max': 0.30},
        "_Pulseclarity_Mean": {'min': 0.01, 'max': 0.12},
        "_Eventdensity_Mean": {'min': 0.01, 'max': 0.20}
    }

# Display feature ranges
st.header('Feature Ranges')
feature_ranges_df = pd.DataFrame(feature_ranges).T
st.dataframe(feature_ranges_df)

# Input for prediction
st.header('Predict Emotion')
input_data = []
for feature in features:
    value = st.number_input(
        f'{feature} (Range: {feature_ranges[feature]["min"]} - {feature_ranges[feature]["max"]})', 
        min_value=float(feature_ranges[feature]['min']), 
        max_value=float(feature_ranges[feature]['max']),
        step=0.01
    )
    input_data.append(value)

# Prediction button
if st.button('Predict'):
    input_data = np.array(input_data).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_emotion = label_encoder.inverse_transform(predicted_class)
    
    st.write(f'Predicted Emotion: **{predicted_emotion[0]}**')
