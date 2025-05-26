# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --- Configuration ---
# Hardcoded credentials for demonstration purposes.
# In a real application, use a secure database or authentication service.
USERS = {
    "user1": "pass123",
    "admin": "admin"
}

# Define paths for the model and scaler
MODEL_PATH = 'fraud_detection_model.pkl'
SCALER_PATH = 'scaler.pkl'

# --- Load Model and Scaler ---
# These will be global variables accessible by the functions below.
model = None
scaler = None

"""
## Fraud Detection Model using Streamlit

### Project Overview

This project demonstrates a basic fraud detection system. It consists of:
-   An authentication page to simulate user login.
-   A machine learning model (RandomForestClassifier) trained on synthetic transaction data.
-   A Streamlit interface for users to input transaction details.
-   Real-time prediction of whether a transaction is fraudulent or legitimate.
-   Basic insights and probability scores.

"""

# --- Authentication Page ---
def show_login_page():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success(f"Welcome, {username}!")
            st.rerun() # Rerun to switch to main app
        else:
            st.sidebar.error("Invalid Username or Password")
            
            
# --- Load Models ---
def load_models():
    global model, scaler
    # Ensure model and scaler files exist before attempting to load
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file '{MODEL_PATH}' not found. "
                 f"Please ensure the training script (train.py) has been run successfully to generate this file.")
        st.stop()
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file '{SCALER_PATH}' not found. "
                 f"Please ensure the training script (train.py) has been run successfully to generate this file.")
        st.stop()
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    # The 'model' and 'scaler' variables are now globally available.
    except Exception as e:
        st.error(f"Fatal error loading model or scaler: {e}. The application cannot start.")
        st.stop()            

# --- Main Application Page ---
def show_main_app():
    st.title(f"Welcome, {st.session_state.username}!")
    st.header("Fraud Detection System")

    st.write("Enter transaction details below to predict if it's fraudulent.")

    # --- User Input Fields ---
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01)
    with col2:
        # Time in seconds (simulated)
        time_seconds = st.slider("Time (seconds from start of data)", 0, 172800, 36000) # Max 48 hours

    st.subheader("Transaction Context (for feature engineering)")
    col3, col4 = st.columns(2)
    with col3:
        hour_of_day = st.slider("Hour of Day (0-23)", 0, 23, 12)
    with col4:
        day_of_week = st.selectbox("Day of Week", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][x])


    if st.button("Predict if Fraudulent"):
        # --- Preprocess User Input ---
        # Create a DataFrame from user input, matching training features
        input_data = pd.DataFrame([[time_seconds, amount, hour_of_day, day_of_week, np.log1p(amount)]],
                                  columns=['Time', 'Amount', 'Hour', 'DayOfWeek', 'Amount_Log'])

        # Scale the input data using the loaded scaler
        scaled_input = scaler.transform(input_data)

        # --- Make Prediction ---
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"🚨 **Predicted as FRAUDULENT!**")
            st.write(f"Probability of Fraud: **{prediction_proba[1]*100:.2f}%**")
            st.warning("This transaction has a high likelihood of being fraudulent. Further investigation is recommended.")
        else:
            st.success(f"✅ **Predicted as LEGITIMATE**")
            st.write(f"Probability of Legitimate: **{prediction_proba[0]*100:.2f}%**")
            st.info("This transaction appears legitimate based on the model's analysis.")

        st.markdown("---")

    # --- Logout Button ---
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun() # Rerun to go back to login page

# --- Main App Logic (Session State Management) ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None

if st.session_state.logged_in:
    load_models()
    show_main_app()
else:
    show_login_page()

