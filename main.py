import streamlit as st
import numpy as np
import pickle

# Load models and encoders
with open('crop_prediction_model.pkl', 'rb') as f:
    crop_model = pickle.load(f)

with open('fertilizer_model.pkl', 'rb') as f:
    fertilizer_model = pickle.load(f)

with open('le_crop.pkl', 'rb') as f:
    le_crop = pickle.load(f)

with open('le_fert.pkl', 'rb') as f:
    le_fert = pickle.load(f)

st.title("🌾 Smart Agri Advisor")

task = st.sidebar.radio("Select Prediction Task:", ["Crop Recommendation", "Fertilizer Recommendation"])

if task == "Crop Recommendation":
    st.header("Crop Recommendation")
    N = st.number_input("Nitrogen", min_value=0.0)
    P = st.number_input("Phosphorus", min_value=0.0)
    pH = st.number_input("pH", min_value=0.0, max_value=14.0)
    Temperature = st.number_input("Temperature", min_value=-10.0, max_value=60.0)
    
    if st.button("Predict Crop"):
        features = np.array([[N, P, pH, Temperature]])
        pred_code = crop_model.predict(features)[0]
        crop = le_crop.inverse_transform([pred_code])[0]
        st.success(f"🌱 Recommended Crop: **{crop}**")

elif task == "Fertilizer Recommendation":
    st.header("Fertilizer Recommendation")
    N = st.number_input("Nitrogen", min_value=0.0)
    P = st.number_input("Phosphorus", min_value=0.0)
    K = st.number_input("Potassium", min_value=0.0)
    pH = st.number_input("pH", min_value=0.0, max_value=14.0)
    Moisture = st.number_input("Moisture", min_value=0.0, max_value=100.0)
    Temperature = st.number_input("Temperature", min_value=-10.0, max_value=60.0)
    
    if st.button("Predict Fertilizer"):
        features = np.array([[N, P, K, pH, Moisture, Temperature]])
        pred_code = fertilizer_model.predict(features)[0]
        st.success(f"🧪 Recommended Fertilizer: **{pred_code}**")
