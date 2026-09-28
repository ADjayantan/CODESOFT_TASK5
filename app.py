import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection App", page_icon="💳")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter the transaction details below to check if it's Genuine or Fraudulent.")

# Load the trained model
@st.cache_resource
def load_model():
    try:
        return joblib.load("fraud_detection_model.joblib")
    except Exception as e:
        return None

model = load_model()

if model is None:
    st.error("Model file 'fraud_detection_model.joblib' not found. Please train the model first.")
else:
    st.sidebar.header("Transaction Features (V1-V28, Time, Amount)")
    st.sidebar.info("Since credit card features are usually PCA transformed to protect privacy (V1 to V28), you can simulate a transaction here.")
    
    # We need 30 features. Time, Amount, V1-V28.
    # Our script trained on scaled features. 
    # But wait, our script scaled them inside preprocess_data before training.
    # To keep this demo simple, we'll just generate 30 random sliders or inputs for the 30 features the model expects.
    
    st.write("### Simulate a Transaction")
    if st.button("Generate Random Genuine Transaction"):
        st.session_state['data'] = np.random.normal(0, 1, 30).reshape(1, -1)
        st.success("Generated features for a normal-looking transaction!")
        
    if st.button("Generate Random Suspicious Transaction"):
        # Shift the distribution to simulate anomalous behavior
        st.session_state['data'] = np.random.normal(3, 2, 30).reshape(1, -1)
        st.error("Generated features for a highly suspicious transaction!")
        
    if 'data' in st.session_state:
        features = st.session_state['data']
        
        st.write("**Feature Vector:**", np.round(features, 2))
        
        if st.button("Predict Fraud"):
            prediction = model.predict(features)
            
            if prediction[0] == 1:
                st.error("⚠️ ALERT: This transaction is predicted as FRAUDULENT!")
            else:
                st.success("✅ This transaction is predicted as GENUINE.")
