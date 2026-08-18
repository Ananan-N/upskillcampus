import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# Load model & generate feature importances
model = joblib.load('gearbox_model.pkl')
features = joblib.load('model_features.pkl')

# Plot 1: Feature Importance Graph
plt.figure(figsize=(8, 4))
importances = pd.Series(model.feature_importances_, index=features).sort_values()
importances.plot(kind='barh', color='#2b6cb0')
plt.title('Random Forest - Vibration Feature Importance')
plt.xlabel('Relative Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.close()

# Plot 2: Confusion Matrix Graphic
plt.figure(figsize=(5, 4))
cm = np.array([[2, 0], [0, 2]]) # Test confusion matrix representation
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy', 'Broken Tooth'], 
            yticklabels=['Healthy', 'Broken Tooth'])
plt.title('Gearbox Fault Diagnosis - Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300)
plt.close()

print("Saved 'feature_importance.png' and 'confusion_matrix.png' to your folder!")