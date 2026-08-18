import os
import glob
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

print("1. Starting feature extraction process...")

# Get current folder path
current_dir = os.getcwd()
print(f"2. Scanning directory: {current_dir}")

# Locate all text files recursively
txt_files = glob.glob(os.path.join(current_dir, "**", "*.txt"), recursive=True)
print(f"3. Found {len(txt_files)} text file(s).")

if len(txt_files) == 0:
    print("ERROR: No .txt files found! Check if 'Healthy Data' and 'BrokenTooth Data' folders exist in this directory.")
    exit()

dataset_rows = []

for file_path in txt_files:
    file_name = os.path.basename(file_path)
    label = 0 if file_name.lower().startswith('h') else 1
    
    try:
        load_val = int(file_name.lower().split('hz')[1].replace('.txt', ''))
    except Exception:
        load_val = 0
        
    # Load continuous 4-channel vibration readings
    df_raw = pd.read_csv(file_path, sep=r'\s+', header=None, names=['s1', 's2', 's3', 's4'])
    
    row = {
        'file_name': file_name,
        'load': load_val,
        's1_std': df_raw['s1'].std(), 's2_std': df_raw['s2'].std(),
        's3_std': df_raw['s3'].std(), 's4_std': df_raw['s4'].std(),
        's1_rms': np.sqrt(np.mean(df_raw['s1']**2)), 's2_rms': np.sqrt(np.mean(df_raw['s2']**2)),
        's3_rms': np.sqrt(np.mean(df_raw['s3']**2)), 's4_rms': np.sqrt(np.mean(df_raw['s4']**2)),
        's1_p2p': df_raw['s1'].max() - df_raw['s1'].min(), 's2_p2p': df_raw['s2'].max() - df_raw['s2'].min(),
        's3_p2p': df_raw['s3'].max() - df_raw['s3'].min(), 's4_p2p': df_raw['s4'].max() - df_raw['s4'].min(),
        'target': label
    }
    dataset_rows.append(row)

df = pd.DataFrame(dataset_rows)
print(f"4. Feature extraction complete! Table shape: {df.shape}")

feature_cols = [c for c in df.columns if c not in ['file_name', 'target']]
X = df[feature_cols]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("5. Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("\n--- MODEL PERFORMANCE ---")
print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Broken Tooth (1)']))

print("6. Saving model artifacts...")
joblib.dump(model, 'gearbox_model.pkl')
joblib.dump(feature_cols, 'model_features.pkl')

print("\nSUCCESS: 'gearbox_model.pkl' and 'model_features.pkl' saved to project folder!")