# 🛒 E-commerce Customer Spending and Satisfaction Level Prediction

This project applies a comprehensive set of machine learning techniques to analyze customer behavior in an e-commerce setting. The goal is to **predict spending habits and satisfaction levels** based on various features derived from customer profiles and interactions.

## 🚀 Key Objectives

* Predict customer spending behavior using regression models.
* Classify customer satisfaction level based on historical activity.
* Perform in-depth data preprocessing, EDA, feature engineering, model evaluation, and explainability analysis.

## 📊 Key Features & Techniques

### 🔍 Data Preprocessing & Feature Engineering

* Handled missing data using KNN and Iterative imputation.
* Encoded categorical variables using One-Hot and Label Encoding.
* Normalized and standardized features for consistent input.

### 📈 Machine Learning Models

**Supervised Learning**

* Linear Regression, Logistic Regression
* SVM, KNN, Naive Bayes, Decision Trees

**Unsupervised Learning**

* Clustering: K-Means, Hierarchical
* Validation: Silhouette Score, Elbow Method

**Ensemble Learning**

* Random Forest, Gradient Boosting, XGBoost, Stacking

**Dimensionality Reduction**

* PCA and LDA to reduce feature space for efficiency.

### 📊 Evaluation Metrics

* Confusion Matrix, ROC-AUC, PR Curves, F1-Score
* MAE, RMSE, R² for regression tasks
* Applied Cross-Validation (k-fold, LOOCV)

### 🤖 Explainable AI

* Used **LIME** and **SHAP** to interpret predictions and identify important features.

### 📉 Imbalanced Data Handling

* Applied **SMOTE**, **ADASYN**, and class weight balancing techniques.

## 📈 Visualizations

* Heatmaps, pair plots, feature importance graphs.
* Clustering visualizations and dimensionality-reduced plots.

## 🛠 Tools and Libraries

* Python (Scikit-learn, Pandas, Numpy, Matplotlib, Seaborn)
* Jupyter Notebooks
* GitHub for version control

## 📂 Folder Structure

```bash
📦project-root
 ┣ 📂data
 ┣ 📂notebooks
 ┣ 📂models
 ┣ 📂visualizations
 ┗ 📄README.md
```

## 📌 Future Work

* Deploy trained model using Flask or Streamlit.
* Build interactive dashboard for business use.

## 🔗 GitHub Repository

https://github.com/zx784/E-commerce-Customer-spending-and-satisfaction-level-prediction
