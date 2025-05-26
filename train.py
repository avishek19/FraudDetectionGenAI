# --- Model Training (Run this part once to generate model and scaler files) ---
import os, random
import streamlit as st

# Define paths for the model and scaler
MODEL_PATH = 'fraud_detection_model.pkl'
SCALER_PATH = 'scaler.pkl'

def generate_amount(num_legit, max_value, min_value):
    amount=[]
    for _ in range(num_legit):
        value = random.uniform(min_value, max_value)
        amount.append(round(value,2))
    return amount

def train_and_save_model():
    st.write("### Training and Saving Model (Run once)")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import joblib

    # 1. Generate Synthetic Data
    np.random.seed(42)
    num_legit = 100000
    num_fraud = 1000

    # Legitimate transactions
    legit_time = np.random.randint(0, 172800, num_legit) # 48 hours in seconds
    legit_amount = generate_amount(num_legit,5500,20)
    legit_class = np.zeros(num_legit)

    # Fraudulent transactions (e.g., higher amounts, specific time windows)
    fraud_time = np.random.randint(0, 86400, num_fraud) # First 24 hours
    fraud_amount = np.random.normal(21000, 100, num_fraud)
    fraud_amount[fraud_amount < 0] = 100 # Ensure positive amounts
    fraud_class = np.ones(num_fraud)

    # Combine
    time = np.concatenate([legit_time, fraud_time])
    amount = np.concatenate([legit_amount, fraud_amount])
    transaction_class = np.concatenate([legit_class, fraud_class])

    df_train = pd.DataFrame({
        'Time': time,
        'Amount': amount,
        'Class': transaction_class
    })
    st.write('Data before formatting:',df_train.head(4)) 

    # 2. Feature Engineering
    df_train['Hour'] = (df_train['Time'] % 86400) // 3600 # Seconds in a day / seconds in an hour
    df_train['DayOfWeek'] = (df_train['Time'] // 86400) % 7 # Seconds in a day
    df_train['Amount_Log'] = np.log1p(df_train['Amount']) # Log transform for skewed data
    st.write('Data after formatting:',df_train.head(4)) 


    # 3. Data Preprocessing (Scaling)
    features = ['Time', 'Amount', 'Hour', 'DayOfWeek', 'Amount_Log']
    X = df_train[features]
    y = df_train['Class']
    st.write('Synthetic Data used for model (rows)',X.head(2))
    df_train.to_csv('synthetic_data.csv', index=False) ## Save Synthetic Data for analysis

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaleData = pd.DataFrame(X_scaled, columns=features) 
    df_scaleData.to_csv('scaled_data.csv', index=False) ## Save Scale Data for analysis

    # 4. Model Training
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    st.write('Train data size:',X_train.shape)
    st.write('Test data size:',X_test.shape)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # Balanced for imbalanced data
    model.fit(X_train, y_train)
    
    # 5. Save Model and Scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    st.success(f"Model saved to {MODEL_PATH} and Scaler saved to {SCALER_PATH}")
    st.write("Training Metrics (on test set):")
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report, confusion_matrix
    st.text(classification_report(y_test, y_pred))
    st.text("Confusion Matrix:")
    st.text(confusion_matrix(y_test, y_pred))
    

# Check if model and scaler exist, if not, train them
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.warning("Model or scaler not found. Training and saving them now...")
    train_and_save_model()
else:
    st.info("Model and scaler found. Skipping training.")

# --- Load Model and Scaler (for the Streamlit App) ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    st.sidebar.success("Model and Scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading model or scaler: {e}. Please ensure '{MODEL_PATH}' and '{SCALER_PATH}' exist.")
    st.stop() # Stop the app if model loading fails
