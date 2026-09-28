import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Advanced Fraud Detection", page_icon="💳", layout="wide")

st.title("💳 Advanced Credit Card Fraud Detection System")
st.markdown("""
Welcome to the interactive **Credit Card Fraud Detection** dashboard for Task 5!
This tool uses a Machine Learning model (**Random Forest**) trained with **SMOTE** on a highly imbalanced dataset to predict if a transaction is **Genuine** or **Fraudulent**.
""")

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
    # Sidebar for Model Insights
    st.sidebar.header("📊 Model Insights")
    st.sidebar.info("The model expects 30 features: **V1-V28 (PCA transformed), Scaled Amount, and Scaled Time**.")
    
    try:
        image = Image.open('fraud_confusion_matrix.png')
        st.sidebar.image(image, caption='Model Confusion Matrix (Random Forest)', use_container_width=True)
    except Exception:
        pass
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to use this app?")
    st.sidebar.markdown("1. Use the simulator buttons to generate a synthetic transaction profile.\n2. Adjust the Transaction Amount and Time manually.\n3. Click **Analyze Transaction** to see the prediction and confidence score!")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("🛠️ Transaction Simulator")
        
        st.write("Generate background PCA features (V1-V28):")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            gen_normal = st.button("🟢 Generate Genuine Profile")
        with col_btn2:
            gen_fraud = st.button("🔴 Generate Suspicious Profile")
            
        if gen_normal:
            st.session_state['v_features'] = np.random.normal(0, 1, 28)
            st.session_state['type'] = "Genuine"
        if gen_fraud:
            st.session_state['v_features'] = np.random.normal(3.5, 2.5, 28)
            st.session_state['type'] = "Suspicious"
            
        st.markdown("---")
        # User inputs for interpretable features
        txn_amount = st.number_input("💳 Transaction Amount ($)", min_value=0.0, value=150.00, step=10.0)
        txn_time = st.number_input("⏱️ Time Elapsed (seconds)", min_value=0, value=3600, step=100)
            
    with col2:
        st.subheader("🔍 Prediction Results")
        if 'v_features' not in st.session_state:
            st.info("👈 Please click one of the 'Generate Profile' buttons on the left to start.")
        else:
            st.write(f"**Loaded Profile:** `{st.session_state['type']} Pattern`")
            
            # Reconstruct the 30 features: V1..V28, scaled_amount, scaled_time
            # We apply a rough mock scaling since the original scaler isn't saved in the joblib.
            scaled_amount = (txn_amount - 100) / 250.0
            scaled_time = (txn_time - 50000) / 40000.0
            
            features = list(st.session_state['v_features']) + [scaled_amount, scaled_time]
            features_array = np.array(features).reshape(1, -1)
            
            st.write("---")
            
            if st.button("🚀 Analyze Transaction", type="primary"):
                with st.spinner("Analyzing transaction patterns..."):
                    prediction = model.predict(features_array)
                    probabilities = model.predict_proba(features_array)[0]
                    
                    fraud_prob = probabilities[1] * 100
                    genuine_prob = probabilities[0] * 100
                    
                    if prediction[0] == 1:
                        st.error("### 🚨 ALERT: FRAUDULENT TRANSACTION DETECTED")
                        st.metric(label="Fraud Probability", value=f"{fraud_prob:.2f}%")
                        st.progress(int(fraud_prob))
                        st.markdown("> **Action Required:** The system has flagged this transaction as highly anomalous based on PCA feature analysis.")
                    else:
                        st.success("### ✅ TRANSACTION APPROVED: GENUINE")
                        st.metric(label="Genuine Probability", value=f"{genuine_prob:.2f}%")
                        st.progress(int(genuine_prob))
                        st.markdown("> **Status:** This transaction matches normal spending patterns and has been approved.")
                        
            with st.expander("Show raw feature vector (V1-V28, Amount, Time)"):
                st.dataframe(pd.DataFrame(features_array, columns=[f"V{i}" for i in range(1, 29)] + ["Scaled_Amount", "Scaled_Time"]))
