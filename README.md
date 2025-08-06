
# 🫀 Heart Disease Prediction using Logistic Regression

This project implements a heart disease prediction model using logistic regression with the help of Scikit-learn. It includes data loading, preprocessing, training, evaluation, and visualization.

---

## 📁 Dataset

- The dataset used is `heart.csv`.
- The dataset contains medical attributes of patients along with a target column indicating the presence of heart disease (`1` for presence, `0` for absence).

### 📊 Features
| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (1 = male; 0 = female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol in mg/dl |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting electrocardiographic results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels (0–3) colored by fluoroscopy |
| thal | Thalassemia (1 = normal; 2 = fixed defect; 3 = reversable defect) |
| target | 1 = presence of heart disease; 0 = absence |

---

## ⚙️ Libraries Used

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
```

---

## 🧹 Data Preprocessing

- Loaded the CSV file into a DataFrame.
- Checked for null values.
- Described dataset statistics.
- Split the dataset into features (`X`) and target (`Y`).
- Performed train-test split with stratification.

```python
data = pd.read_csv('heart.csv')
X = data.drop(columns='target', axis=1)
Y = data['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

---

## 🧠 Model Training

- Used `LogisticRegression` from Scikit-learn.
- Trained the model on `X_train`.

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

---

## 📈 Model Evaluation

- Calculated accuracy on both training and test sets.
- Printed classification report.
- Displayed confusion matrix.

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X_test_prediction = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, X_test_prediction))
print(classification_report(Y_test, X_test_prediction))

# Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(Y_test, X_test_prediction), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

## 📊 Visualizations

### 1. Target Distribution

```python
sns.countplot(x='target', data=data)
plt.title('Target Class Distribution')
plt.show()
```

### 2. Correlation Heatmap

```python
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()
```

### 3. Feature Importance

```python
importance = pd.Series(model.coef_[0], index=X.columns)
importance.sort_values().plot(kind='barh')
plt.title('Feature Importance (Logistic Regression)')
plt.show()
```

### 4. ROC Curve

```python
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(Y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(Y_test, y_prob):.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 🔍 Prediction for Custom Input

```python
input_data = (58,0,0,100,248,0,0,122,0,1,1,0,2)
input_array = np.asarray(input_data).reshape(1, -1)
prediction = model.predict(input_array)

if prediction[0] == 1:
    print("The person has heart disease.")
else:
    print("The person does not have heart disease.")
```

---

## 🧩 Future Enhancements

- Try other models: RandomForest, XGBoost, SVM, etc.
- Perform hyperparameter tuning.
- Use cross-validation for better generalization.
- Integrate a front-end UI for real-time predictions.

---

## 👤 Author Information

- **Project by:** `Ruhi Tahmidul Islam`
- **Model:** `Logistic Regression`
- **Topic:** `Medical diagnosis – heart disease detection`
