
# 🧠 AI/ML Engineering Internship Tasks

This repository contains all tasks completed as part of my **AI/ML Engineering Internship** at **DevelopersHub Corporation**. Each task showcases a different machine learning use case, including regression, classification, and time series prediction.

---

## 📁 Tasks Overview

| Task No. | Task Name                       | Model Used                      | Type           | Output                  |
|----------|----------------------------------|----------------------------------|----------------|--------------------------|
| Task 2   | Predict Future Stock Prices      | Linear Regression / Random Forest | Regression     | Next day's stock price   |
| Task 3   | Heart Disease Prediction         | Logistic Regression / Random Forest | Classification | Disease Risk (Yes/No)    |
| Task 6   | House Price Prediction           | Linear Regression               | Regression     | House Price (PKR)       |

---

## 📈 Task 2: Predict Future Stock Prices

### 🎯 Objective:
Use historical data from Yahoo Finance to predict the **next day’s closing price** for a selected stock (e.g., Apple, Tesla).

### 📚 Dataset:
- Retrieved using `yfinance` Python library.
- Features: `Open`, `High`, `Low`, `Volume`, `Daily Return`, `MA_7`, `MA_21`, `Volatility`

### 📦 Tools:
- `pandas`, `scikit-learn`, `matplotlib`, `yfinance`

### 🧠 Model:
- Linear Regression
- Random Forest Regressor

### 📊 Output:
- Actual vs Predicted closing price plot
- Saved model: `stock_price_model.pkl`, `scaler.pkl`

---

## ❤️ Task 3: Heart Disease Prediction

### 🎯 Objective:
Predict whether a patient is at **risk of heart disease** using health indicators.

### 📚 Dataset:
- `heart.csv` (UCI Heart Disease Dataset)
- Features: age, sex, chest pain type, cholesterol, resting BP, fasting blood sugar, etc.

### 📦 Tools:
- `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### 🧠 Model:
- Logistic Regression
- Random Forest Classifier

### ✅ Evaluation:
- Accuracy, Precision, Recall, Confusion Matrix, ROC-AUC
- Saved model: `heart_model.pkl`

---

## 🏠 Task 6: House Price Prediction

### 🎯 Objective:
Estimate **property prices** based on features like area, location, bedrooms, bathrooms, etc.

### 📚 Dataset:
- `house_data.csv` (Kaggle Real Estate Dataset)

### 📦 Features:
- `Area in Marla`, `bedroom`, `bathroom`, `Property_type`, `City`, `Location`, `purpose`
- Target: `Price`

### 🧠 Model:
- Linear Regression
- Feature Scaling with `StandardScaler`

### 📊 Evaluation:
| Metric | Value |
|--------|-------|
| MAE    | ~350,000 PKR |
| RMSE   | ~550,000 PKR |
| R²     | ~0.78         |

### 💾 Output:
- Saved model: `house_price_model.pkl`
- Saved scaler: `scaler.pkl`

---

## 🛠️ How to Run

```bash
# Clone the repository
git clone https://github.com/Abdul-Wahab1010/AI-ML-Engineering-Internship-Tasks.git
cd AI-ML-Engineering-Internship-Tasks

# Enter any task folder
cd Task_02-Predict-Future-Stock-Prices

# Install requirements
pip install pandas scikit-learn matplotlib yfinance seaborn

# Open notebook
jupyter notebook Task2_StockPrice.ipynb
```

---

## 👤 Author

**Abdul Wahab**  
AI/ML Engineering Intern – DevelopersHub Corporation  
GitHub: [@Abdul-Wahab1010](https://github.com/Abdul-Wahab1010)

---

### ✅ All models and data are included for testing and reproducibility.
