import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Gearbox Predictive Maintenance", page_icon="⚙️", layout="wide")

st.title("⚙️ Drivetrain Gearbox Fault Diagnosis System")
st.markdown("##### Real-Time Predictive Maintenance Dashboard using 4-Channel Vibration Sensors")
st.write("---")

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('gearbox_model.pkl')
        features = joblib.load('model_features.pkl')
        return model, features
    except Exception:
        return None, None

model, expected_features = load_model_artifacts()

if model is None:
    st.error("Model files not found! Please run 'train_model.py' first.")
else:
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("1. Input Operational Parameters")
        load_percentage = st.slider("Select Dynamometer Load Percentage (%)", 0, 90, 30, step=10)
        uploaded_file = st.file_uploader("Upload Raw Sensor Recording (.txt)", type=["txt"])

    with col2:
        st.subheader("2. Diagnostic Analysis Output")
        
        if uploaded_file is not None:
            # Load continuous raw 4-channel readings
            df_raw = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=['s1', 's2', 's3', 's4'])
            
            # Feature extraction on-the-fly for uploaded file
            input_features = pd.DataFrame([{
                'load': load_percentage,
                's1_std': df_raw['s1'].std(), 's2_std': df_raw['s2'].std(),
                's3_std': df_raw['s3'].std(), 's4_std': df_raw['s4'].std(),
                's1_rms': np.sqrt(np.mean(df_raw['s1']**2)), 's2_rms': np.sqrt(np.mean(df_raw['s2']**2)),
                's3_rms': np.sqrt(np.mean(df_raw['s3']**2)), 's4_rms': np.sqrt(np.mean(df_raw['s4']**2)),
                's1_p2p': df_raw['s1'].max() - df_raw['s1'].min(), 's2_p2p': df_raw['s2'].max() - df_raw['s2'].min(),
                's3_p2p': df_raw['s3'].max() - df_raw['s3'].min(), 's4_p2p': df_raw['s4'].max() - df_raw['s4'].min(),
            }])[expected_features]

            prediction = model.predict(input_features)[0]
            confidence = model.predict_proba(input_features)[0]

            if prediction == 1:
                st.error("DIAGNOSIS: BROKEN TOOTH FAULT DETECTED")
                st.metric("Model Confidence", f"{confidence[1]*100:.1f}%")
                st.warning("**Recommended Action:** Issue work order for mechanical gearbox inspection.")
            else:
                st.success("DIAGNOSIS: GEARBOX HEALTHY")
                st.metric("Model Confidence", f"{confidence[0]*100:.1f}%")
                st.info("**Recommended Action:** Gearbox operating normally within safe thresholds.")

            #visualisation
            st.write("---")
            st.subheader("Raw 4-Channel Vibration Signals (First 1,000 Samples)")
            
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(df_raw['s1'][:1000], label='Sensor 1 (s1)', alpha=0.8)
            ax.plot(df_raw['s2'][:1000], label='Sensor 2 (s2)', alpha=0.8)
            ax.plot(df_raw['s3'][:1000], label='Sensor 3 (s3)', alpha=0.8)
            ax.plot(df_raw['s4'][:1000], label='Sensor 4 (s4)', alpha=0.8)
            ax.set_xlabel("Sample Point Index")
            ax.set_ylabel("Vibration Amplitude")
            ax.legend(loc="upper right")
            st.pyplot(fig)

            st.subheader("Computed RMS Energy per Channel")
            fig2, ax2 = plt.subplots(figsize=(8, 3))
            rms_vals = [input_features['s1_rms'][0], input_features['s2_rms'][0], 
                        input_features['s3_rms'][0], input_features['s4_rms'][0]]
            ax2.bar(['Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4'], rms_vals, color='#3182ce')
            ax2.set_ylabel("RMS Vibration Energy")
            st.pyplot(fig2)

            # previewing data table 
            st.write("---")
            st.subheader("Raw Sensor Signals Preview (`s1`, `s2`, `s3`, `s4`)")
            st.dataframe(df_raw.head(20))

        else:
            st.info("Upload a `.txt` vibration file on the left panel to trigger automatic analysis.")