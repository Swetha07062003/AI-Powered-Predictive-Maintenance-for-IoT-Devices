# ===============================
# AI Predictive Maintenance System
# FINAL CLEAN VERSION
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ===============================
# 1. LOAD DATASET
# ===============================
print("Loading dataset...")

file_path = "data/train_FD001.txt"

columns = ['engine_id', 'cycle'] + \
          [f'op{i}' for i in range(1, 4)] + \
          [f'sensor_{i}' for i in range(1, 22)]

data = pd.read_csv(file_path, sep=' ', header=None)
data = data.dropna(axis=1)
data.columns = columns

print("Dataset Loaded Successfully!")
print(data.head())

# ===============================
# 2. CALCULATE RUL
# ===============================
print("\nCalculating RUL...")

data['RUL'] = data.groupby('engine_id')['cycle'].transform('max') - data['cycle']

print("RUL added!")

# ===============================
# 3. CREATE FAILURE LABEL
# ===============================
print("\nCreating failure labels...")

threshold = 30
data['failure'] = data['RUL'].apply(lambda x: 1 if x <= threshold else 0)

print(data[['engine_id', 'cycle', 'RUL', 'failure']].head())

# ===============================
# 4. FEATURE SELECTION
# ===============================
features = [col for col in data.columns if 'sensor' in col]
X = data[features]
y = data['failure']

# ===============================
# 5. TRAIN TEST SPLIT
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6. MODEL TRAINING
# ===============================
print("\nTraining model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model Training Complete!")

# ===============================
# 7. EVALUATION
# ===============================
print("\nEvaluating model...")

y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ===============================
# 8. SAVE MODEL
# ===============================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/predictive_model.pkl")

print("\nModel saved successfully!")

# ===============================
# 9. CREATE OUTPUT FOLDER
# ===============================
os.makedirs("outputs", exist_ok=True)

# ===============================
# 10. CONFUSION MATRIX GRAPH
# ===============================
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

plt.xticks([0, 1], ["No Failure", "Failure"])
plt.yticks([0, 1], ["No Failure", "Failure"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.close()

# ===============================
# 11. FAILURE DISTRIBUTION
# ===============================
plt.figure()
data['failure'].value_counts().plot(kind='bar')

plt.title("Failure Distribution")
plt.xlabel("Failure (0 = No, 1 = Yes)")
plt.ylabel("Count")

plt.savefig("outputs/failure_distribution.png")
plt.close()

# ===============================
# 12. RUL GRAPH (FIRST 5 ENGINES)
# ===============================
plt.figure()

for i in range(1, 6):
    temp = data[data['engine_id'] == i]
    plt.plot(temp['cycle'], temp['RUL'], label=f'Engine {i}')

plt.title("RUL vs Cycle (First 5 Engines)")
plt.xlabel("Cycle")
plt.ylabel("Remaining Useful Life (RUL)")
plt.legend()

plt.savefig("outputs/rul_5_engines.png")
plt.close()

# ===============================
# 13. ACTUAL VS PREDICTED GRAPH
# ===============================
plt.figure()

sample_size = 100
plt.plot(range(sample_size), y_test.values[:sample_size], label="Actual")
plt.plot(range(sample_size), y_pred[:sample_size], label="Predicted")

plt.title("Actual vs Predicted Failure")
plt.xlabel("Sample Index")
plt.ylabel("Failure (0/1)")
plt.legend()

plt.savefig("outputs/actual_vs_predicted.png")
plt.close()

# ===============================
# 14. FAILURE PROBABILITY GRAPH
# ===============================
plt.figure()

y_prob = model.predict_proba(X_test)[:, 1]

plt.plot(y_prob[:200])
plt.title("Failure Prediction Probability")
plt.xlabel("Sample Index")
plt.ylabel("Probability")

plt.savefig("outputs/failure_probability.png")
plt.close()

# ===============================
# 15. FINAL OUTPUT
# ===============================
print("\nAll graphs saved in 'outputs' folder!")

print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")