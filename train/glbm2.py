import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

input_csv = "seleksi_fitur\selected_glcm_features.csv"
df = pd.read_csv(input_csv)

X = df.drop(columns=['class'])  
y = df['class']                 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    class_weight='balanced'  
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("\n=== Training Summary ===")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(matrix)

output_dir = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/models"
os.makedirs(output_dir, exist_ok=True)
model_path = os.path.join(output_dir, "lgbm_glcm_model.pkl")
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")