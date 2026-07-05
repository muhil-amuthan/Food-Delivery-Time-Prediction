import streamlit as st
import pandas as pd
import joblib
import os

# ============================================================
# PAGE CONFIG (must be first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍕",
    layout="centered"
)

# ============================================================
# LOAD MODEL & ENCODERS
# ============================================================
MODEL_PATH = "delivery_model.pkl"
ENCODERS_PATH = "encoders.pkl"

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and LabelEncoders."""
    if not os.path.exists(MODEL_PATH):
        return None, None, f"Model file '{MODEL_PATH}' not found."
    if not os.path.exists(ENCODERS_PATH):
        return None, None, f"Encoders file '{ENCODERS_PATH}' not found."
    
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    return model, encoders, None

model, encoders, error_msg = load_model_and_encoders()

if error_msg:
    st.error(f"❌ {error_msg}")
    st.info("""
    **To fix this:**
    1. Run your training script first to generate the model
    2. Add these lines at the end of your training script:
    ```python
    import joblib
    joblib.dump(best_model, "delivery_model.pkl")
    joblib.dump(encoders, "encoders.pkl")
    ```
    """)
    st.stop()

st.success("✅ Model and encoders loaded successfully!")

# ============================================================
# UI
# ============================================================
st.title("🍔 Food Delivery Time Prediction")
st.write("Enter the delivery details below to predict the delivery time.")

# --- Inputs (NO Order_ID — it was dropped during training) ---
col1, col2 = st.columns(2)

with col1:
    distance = st.number_input("Distance (km)", min_value=0.0, value=5.0, step=0.1)
    weather = st.selectbox("Weather", ["Clear", "Foggy", "Rainy", "Windy"])
    traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

with col2:
    vehicle = st.selectbox("Vehicle Type", ["Bike", "Scooter"])
    prep_time = st.number_input("Preparation Time (minutes)", min_value=1, value=15)
    experience = st.number_input("Courier Experience (Years)", min_value=0, value=2)

# ============================================================
# PREDICTION
# ============================================================
if st.button("Predict Delivery Time", type="primary"):
    # Build input DataFrame with EXACT column names from training
    input_data = pd.DataFrame({
        "Distance_km": [distance],
        "Weather": [weather],
        "Traffic_Level": [traffic],
        "Time_of_Day": [time_of_day],
        "Vehicle_Type": [vehicle],
        "Preparation_Time_min": [prep_time],
        "Courier_Experience_yrs": [experience]
    })
    
    # Apply the SAME LabelEncoders used during training
    for col, le in encoders.items():
        if col in input_data.columns:
            # Handle unseen categories gracefully
            try:
                input_data[col] = le.transform(input_data[col])
            except ValueError as e:
                st.error(f"❌ Invalid value for '{col}': {input_data[col].values[0]!r}")
                st.stop()
    
    # Ensure column order matches training exactly
    feature_columns = [
        "Distance_km", "Weather", "Traffic_Level", "Time_of_Day",
        "Vehicle_Type", "Preparation_Time_min", "Courier_Experience_yrs"
    ]
    input_data = input_data[feature_columns]
    
    # Predict
    try:
        prediction = model.predict(input_data)
        st.success(f"### 🕐 Estimated Delivery Time: **{prediction[0]:.2f} minutes**")
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")