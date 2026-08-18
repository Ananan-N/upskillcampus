# upskillcampus
Data Science &amp; Machine Learning Internship

# Predictive Maintenance of Gearbox Using Vibration Sensors

This project delivers an end-to-end Machine Learning pipeline and interactive analytical dashboard for predicting mechanical gearbox failures (specifically Broken Tooth Faults) using multi-channel time-series vibration sensor data.

## 🚀 Key Features

* **Multi-Channel Processing:** Processes 4-channel high-frequency accelerometer signal data across varying dynamometer operational loads (0% to 90%).
* **Feature Extraction:** Extracts key statistical indicators in the time domain, including Root Mean Square (RMS), Standard Deviation (STD), and Peak-to-Peak (P2P) amplitudes per channel.
* **Supervised Classification:** Employs a trained Random Forest model (`gearbox_model.pkl`) to accurately detect Healthy vs. Broken Tooth states.
* **Interactive Dashboard:** Deploys a Streamlit web interface (`app.py`) for uploading raw vibration data files, visualizing dynamic signal waveforms, analyzing RMS energy distributions, and receiving instant fault alerts.

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Machine Learning & Data Processing:** Scikit-Learn, Pandas, NumPy, Joblib
* **Visualization & Web Interface:** Streamlit, Matplotlib, Seaborn

## 📂 Project Structure

```text
upskillcampus/
├── app.py                      # Main Streamlit web application dashboard
├── train_model.py              # Script to process datasets and train the classifier
├── plot_evaluation.py          # Script for feature importance & model metrics visualization
├── requirements.txt            # Python dependencies
├── gearbox_model.pkl           # Trained Random Forest machine learning model
├── model_features.pkl          # Feature names artifact
├── confusion_matrix.png        # Model evaluation artifact
├── feature_importance.png      # Feature importance chart
├── BrokenTooth Data/           # Vibration data under broken tooth fault conditions
├── Healthy Data/               # Vibration data under healthy conditions
└── PredictiveMaintenance_Ananya_USC_UCT.pdf  # Final project report PDF


## 🔧 Installation & Running Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ananan-N/upskillcampus.git](https://github.com/Ananan-N/upskillcampus.git)
   cd upskillcampus

pip install -r requirements.txt
streamlit run app.py
