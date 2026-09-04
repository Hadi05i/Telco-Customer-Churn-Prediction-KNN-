# Telco Customer Churn Prediction (KNN)

A clean Machine Learning workflow using **K-Nearest Neighbors (KNN)** to predict customer churn based on account attributes and tenure.

---

## Overview
* **Goal:** Classify whether a customer will churn (`Yes`/`No`).
* **Selected Features:** `MonthlyCharges`, `tenure`, `PaymentMethod`, and `Contract`.
* **Preprocessing:** 
  * Encoded categorical features with `pd.get_dummies(drop_first=True)`.
  * Scaled numeric values via `StandardScaler` to ensure balanced distance calculation in KNN.
* **Evaluation:** Explored $K$ values from $1$ to $40$ to optimize classification accuracy and plotted the decision curve.

---

## Tech Stack
* Python
* Pandas
* Scikit-Learn
* Matplotlib

---

## Workflow & Code Steps
1. **Data Preparation:** Extracted core behavioral and billing features from the Telco dataset.
2. **Train/Test Split:** 80% training and 20% testing with a fixed random state.
3. **Feature Scaling:** Applied `StandardScaler` fit on train and transformed on test data.
4. **Hyperparameter Tuning:** Iterated through 40 neighbors to observe model variance/bias and plotted the accuracy trend.

---

## How to Run

```bash
# Clone the repo
git clone [https://github.com/](https://github.com/)<your-username>/customer-churn-knn.git

# Install requirements
pip install pandas scikit-learn matplotlib

# Run the script
python main.py
