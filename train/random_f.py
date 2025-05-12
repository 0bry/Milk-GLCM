import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib  
import io
import contextlib

file_path = 'seleksi_fitur\selected_glcm_features.csv'  
data = pd.read_csv(file_path)

X = data.drop('class', axis=1)
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

model = RandomForestClassifier(n_estimators=100, random_state=42)  

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump(model, 'random_forest_model.pkl')
print("Model saved as random_forest_model.pkl")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)  
cm = confusion_matrix(y_test, y_pred)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y)))
plt.xticks(tick_marks, np.unique(y))
plt.yticks(tick_marks, np.unique(y))

thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             ha="center", va="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

plt.subplot(1, 2, 2)  
report = classification_report(y_test, y_pred, output_dict=True)
metrics = pd.DataFrame(report).iloc[:-1, :].T  
metrics.plot(kind='bar', ax=plt.gca())
plt.title('Classification Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()