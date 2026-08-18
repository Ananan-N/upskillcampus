# Predictive Maintenance of Gearbox Using Vibration Sensors

An end-to-end **Machine Learning pipeline and dashboard** for predicting mechanical gearbox failures—specifically **Broken Tooth Faults**—using multi-channel time-series vibration sensor data.

## 🚀 Key Features

* **Multi-Channel Processing:** Processes 4-channel, high-frequency accelerometer signal data collected under varying dynamometer operational loads from **0% to 90%**.
* **Feature Extraction:** Extracts statistical features from the time domain, including:

  * Root Mean Square (RMS)
  * Standard Deviation (STD)
  * Peak-to-Peak (P2P) amplitude
* **Supervised Classification:** Uses a trained **Random Forest** classifier to distinguish between **Healthy** and **Broken Tooth** gearbox conditions.
* **Interactive Dashboard:** Provides a Streamlit-based interface for:

  * Uploading raw vibration data
  * Visualizing signal waveforms
  * Analyzing RMS energy distributions
  * Receiving real-time gearbox fault predictions and alerts

## 🛠️ Tech Stack

| Category                 | Technologies         |
| ------------------------ | -------------------- |
| **Programming Language** | Python 3.x           |
| **Data Processing**      | Pandas, NumPy        |
| **Machine Learning**     | Scikit-learn, Joblib |
| **Visualization**        | Matplotlib, Seaborn  |
| **Web Interface**        | Streamlit            |

## 📂 Project Structure

```text
upskillcampus/
│
├── app.py                         # Main Streamlit dashboard
├── train_model.py                # Dataset processing and model training
├── plot_evaluation.py            # Model evaluation and visualization
├── requirements.txt               # Python dependencies
│
├── gearbox_model.pkl              # Trained Random Forest model
├── model_features.pkl             # Saved model feature names
├── confusion_matrix.png           # Confusion matrix visualization
├── feature_importance.png         # Feature importance visualization
│
├── BrokenTooth Data/              # Vibration data for broken tooth faults
├── Healthy Data/                  # Vibration data for healthy gearboxes
│
└── PredictiveMaintenance_Ananya_USC_UCT.pdf
                                   # Final project report
```

## ⚙️ Installation & Running Locally

### 1. Clone the Repository

```bash
git clone https://github.com/Ananan-N/upskillcampus.git
cd upskillcampus
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will start locally, and Streamlit will provide a URL to open the dashboard in your browser.

## 📊 Model

The project uses a **Random Forest Classifier** trained on statistical features extracted from multi-channel vibration signals.

The model predicts two gearbox conditions:

* **Healthy**
* **Broken Tooth**

The trained model is stored in:

```text
gearbox_model.pkl
```

The corresponding feature names are stored in:

```text
model_features.pkl
```

## 📈 Evaluation

The model evaluation artifacts generated during training include:

* **Confusion Matrix:** `confusion_matrix.png`
* **Feature Importance:** `feature_importance.png`

These visualizations provide insights into the model's classification performance and the contribution of individual vibration features.

## 📁 Dataset

The dataset contains multi-channel vibration measurements recorded under different operational loads.

The project includes data for:

* **Healthy gearbox conditions**
* **Broken tooth fault conditions**

Data is organized into the following directories:

```text
BrokenTooth Data/
Healthy Data/
```

## 📄 Project Report

The complete project documentation and methodology are available in:

```text
PredictiveMaintenance_Ananya_USC_UCT.pdf
```

## 👩‍💻 Author

**Ananya**

Developed as part of the **UpskillCampus Data Science & Machine Learning Internship**.
