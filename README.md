# E-commerce Customer Behavior: Predictive Modeling for Spending & Satisfaction

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0.2-orange.svg)
![Pandas](https://img.shields.io/badge/pandas-1.4.2-blueviolet.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.11.2-green.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains a comprehensive data science project focused on analyzing and predicting e-commerce customer behavior. The goal is to leverage customer demographic and transactional data to forecast two key metrics: **Spending Score** and **Satisfaction Level**. By understanding the factors that drive these outcomes, e-commerce businesses can tailor marketing strategies, improve customer experience, and ultimately boost revenue.

## 📋 Table of Contents
- [Project Vision](#-project-vision)
- [The Dataset](#-the-dataset)
- [Analytical Approach](#-analytical-approach)
- [Modeling & Results](#-modeling--results)
- [Actionable Insights](#-actionable-insights)
- [How to Run This Project](#-how-to-run-this-project)
- [Author](#-author)
- [License](#-license)

## 🎯 Project Vision

In the competitive e-commerce landscape, understanding customers is paramount. This project moves beyond simple historical analysis to build predictive models that answer two critical questions:
1.  **How much is a customer likely to spend?** (Regression task)
2.  **How satisfied is a customer likely to be?** (Classification task)

The insights from these models can empower businesses to segment customers effectively, personalize offers, and proactively address issues to prevent customer churn.

## 📊 The Dataset

The analysis is based on a synthetic e-commerce customer dataset containing a rich mix of demographic and behavioral features.

- **Dataset:** `E-commerce Customer Behavior - Sheet1.csv`
- **Key Features:**
    - **Demographics:** `Gender`, `Age`, `City`, `Membership Type`
    - **Transactional:** `Total Spend`, `Items Purchased`, `Discount Applied`
    - **Behavioral:** `Average Rating`, `Days Since Last Purchase`
- **Target Variables:**
    - `Spending Score`: A score from 1 to 5 indicating a customer's spending propensity.
    - `Satisfaction Level`: A score from 1 to 5 indicating a customer's satisfaction.

## ⚙️ Analytical Approach

A multi-stage methodology was employed to ensure robust and reliable findings.

### 1. Exploratory Data Analysis (EDA)
A deep dive into the data was conducted to uncover initial patterns and relationships.
- **Distributions:** Analyzed the distribution of age, gender, city, and membership types.
- **Correlations:** A heatmap was used to identify correlations between numerical features like `Total Spend` and `Items Purchased`.
- **Behavioral Patterns:** Investigated how spending and satisfaction vary across different segments (e.g., Gold vs. Silver members, customers from different cities).

### 2. Data Preprocessing
The raw data was transformed to prepare it for machine learning.
- **Categorical Encoding:** Features like `Gender`, `City`, and `Membership Type` were converted into a numerical format using one-hot encoding.
- **Feature Scaling:** All numerical features were standardized using `StandardScaler` to ensure that no single feature disproportionately influences the model's predictions.

## 📈 Modeling & Results

Two separate predictive modeling tasks were performed. For each task, multiple algorithms were trained and evaluated to identify the best-performing model.

### Task 1: Predicting Spending Score (Regression)
The goal was to predict the continuous `Spending Score` (1-5).

- **Models Trained:** Linear Regression, Random Forest Regressor, Gradient Boosting Regressor.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² Score.
- **Top Performer:** The **Random Forest Regressor** demonstrated the highest accuracy, effectively capturing the non-linear relationships between customer attributes and their spending behavior.

### Task 2: Predicting Satisfaction Level (Classification)
The goal was to classify customers into one of the five `Satisfaction Level` categories.

- **Models Trained:** Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier.
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- **Top Performer:** The **Random Forest Classifier** again achieved the best results, showing a strong ability to correctly classify customer satisfaction based on their profile and history.

## 💡 Actionable Insights

This analysis yielded several key insights that can inform business strategy:
- **Membership Matters:** `Membership Type` is a powerful predictor of both spending and satisfaction. "Gold" members consistently show higher spending and satisfaction, highlighting the success of premium loyalty programs.
- **Spending Habits are Key:** Features like `Total Spend` and `Items Purchased` are highly correlated with the `Spending Score`, indicating that past behavior is a strong indicator of future spending.
- **Random Forest Excels:** The Random Forest algorithm proved to be the most versatile and accurate model for both regression and classification tasks in this context, making it a reliable choice for deployment.

## 🚀 How to Run This Project

To explore the analysis and run the models yourself, follow these instructions:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/zx784/E-commerce-Customer-spending-and-satisfaction-level-prediction.git](https://github.com/zx784/E-commerce-Customer-spending-and-satisfaction-level-prediction.git)
    cd E-commerce-Customer-spending-and-satisfaction-level-prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    A `requirements.txt` file should be created with the following content:
    ```
    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    jupyternotebook
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Launch the Jupyter Notebook:**
    ```bash
    jupyter notebook "E-commerce Customer spending and satisfaction level prediction.ipynb"
    ```

## ✍️ Author

- **Amro Shiek**

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
