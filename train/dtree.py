import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

df = pd.read_csv('seleksi_fitur\selected_glcm_features.csv')

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", cm)

model_filename = 'decision_tree_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

if report and 'weighted avg' in report:
    classes = list(report.keys())[:-3]
    f1_scores = [report[cls]['f1-score'] for cls in classes]

    plt.figure(figsize=(8, 6))
    plt.bar(classes, f1_scores, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title('F1-Score per Class')
    plt.ylim([0, 1.1])
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
elif report and 'f1-score' in report:
    f1 = report.get('f1-score')
    if f1 is not None:
        plt.figure(figsize=(6, 4))
        plt.bar(['Class 0' if len(y_test.unique()) == 1 and y_test.unique()[0] == 0 else 'Class 1'], [f1], color='skyblue')
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.title('F1-Score')
        plt.ylim([0, 1.1])
        plt.tight_layout()
        plt.show()
    else:
        print("Could not extract F1-score for plotting.")
else:
    print("Classification report is empty or in an unexpected format for plotting.")

plt.figure(figsize=(6, 4))
plt.bar(['Accuracy'], [accuracy], color='lightcoral')
plt.ylabel('Score')
plt.title('Model Accuracy')
plt.ylim([0, 1.1])
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=sorted(y.unique()), yticklabels=sorted(y.unique()))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()