# ❤️ Heart Disease Prediction - Machine Learning Project

This project uses machine learning to predict the presence of heart disease based on patient medical data. It demonstrates the end-to-end ML pipeline: from data preprocessing and model training to evaluation and visualization.

---

## 🚀 Project Highlights

- Clean and well-commented Jupyter workflow  
- Preprocessing, exploratory data analysis (EDA), and feature importance  
- Logistic Regression, Random Forest & Hyperparameter Tuning  
- ROC Curve, Confusion Matrix, and Cross-Validation  
- Model evaluation with Accuracy, Precision, Recall, and F1 Score  

---

## 🧠 Problem Statement

Given a dataset of patients with various health-related attributes, predict whether a patient has heart disease (`target = 1`) or not (`target = 0`).

---

## 📁 Dataset

The dataset used is a cleaned version of the UCI Heart Disease dataset.  
Common features include:

- `age`
- `sex`
- `cp` (chest pain type)
- `trestbps` (resting blood pressure)
- `chol` (cholesterol)
- `fbs` (fasting blood sugar)
- `thalach` (max heart rate)
- `exang` (exercise-induced angina)
- `target` (1 = disease, 0 = no disease)

---

## 🛠️ Technologies & Libraries

- Python 🐍  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- scikit-learn  
  - `LogisticRegression`, `RandomForestClassifier`  
  - `GridSearchCV`, `RandomizedSearchCV`  
  - `confusion_matrix`, `roc_auc_score`, `cross_val_score`

---

## 📊 Model Building Workflow

1. **Data Loading & Cleaning**
2. **Exploratory Data Analysis (EDA)**  
   - Correlation heatmaps  
   - Feature distributions  
   - Target balance
3. **Model Training**
   - Logistic Regression  
   - Random Forest Classifier
4. **Hyperparameter Tuning**
   - `RandomizedSearchCV` for Random Forest  
   - `GridSearchCV` for Logistic Regression
5. **Evaluation**
   - Accuracy, Precision, Recall, F1-score  
   - ROC AUC Score  
   - Confusion Matrix  
   - Cross-validation scores (`cross_val_score`)  

---



