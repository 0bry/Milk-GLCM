import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os


features_path = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/ekstraksi_fitur/glcm_features_ex.csv"
df = pd.read_csv(features_path)

X = df.drop('class', axis=1)  
y = df['class']             

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(le.classes_),
    boosting_type='gbdt',
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          eval_metric='multi_logloss',  
          callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=20)])

y_pred = model.predict(X_test)

print("\n=== Evaluation Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.inverse_transform(range(len(le.classes_)))))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

output_dir = "C:/Users/ASUS VivoBook/OneDrive/Desktop/project_milk/Milk-GLCM/models"
os.makedirs(output_dir, exist_ok=True)

model_path = os.path.join(output_dir, 'glcm_lgbm_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

encoder_path = os.path.join(output_dir, 'label_encoder.pkl')
with open(encoder_path, 'wb') as f:
    pickle.dump(le, f)

print(f"\nModel saved to {model_path}")
print(f"Label encoder saved to {encoder_path}")