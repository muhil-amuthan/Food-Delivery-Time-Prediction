# 🍕 Food Delivery Time Prediction Using Machine Learning

## 📌 Project Overview

This project predicts the estimated food delivery time using Machine Learning techniques. It analyzes various factors such as distance, weather, traffic conditions, preparation time, courier experience, and vehicle type to estimate the delivery time accurately.

The project compares the performance of **Linear Regression** and **Decision Tree Regression** models and includes a **Streamlit web application** for real-time predictions.

---

## 🎯 Objectives

- Collect and preprocess the food delivery dataset.
- Perform Exploratory Data Analysis (EDA).
- Train a Linear Regression model.
- Train a Decision Tree Regression model.
- Compare model performance using evaluation metrics.
- Predict delivery time for new user inputs.
- Build an interactive Streamlit application.

---

## 📂 Project Structure

```text
Food_Delivery_Time_Prediction/
│
├── Food_Delivery_Times.csv
├── Food_Delivery_Prediction.py
├── app.py
├── delivery_model.pkl
├── encoders.pkl
├── requirements.txt
├── README.md
├── Project_Report.pdf
└── screenshots/
```

---

## 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| Order_ID | Unique Order ID |
| Distance_km | Distance between restaurant and customer |
| Weather | Weather condition |
| Traffic_Level | Traffic intensity |
| Time_of_Day | Morning, Afternoon, Evening, Night |
| Vehicle_Type | Delivery vehicle used |
| Preparation_Time_min | Food preparation time |
| Courier_Experience_yrs | Courier experience in years |
| Delivery_Time_min | Target variable |

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

---

## 🤖 Machine Learning Models

### 1. Linear Regression
A regression algorithm that predicts delivery time based on a linear relationship between input features and the target variable.

### 2. Decision Tree Regression
A tree-based machine learning algorithm capable of learning complex relationships between the input features and delivery time.

---

## 📈 Evaluation Metrics

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 📊 Workflow

1. Load Dataset
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Encoding
5. Train-Test Split
6. Train Linear Regression
7. Train Decision Tree Regression
8. Model Evaluation
9. Model Comparison
10. Delivery Time Prediction
11. Streamlit Web Application

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/muhil-amuthan/Food-Delivery-Time-Prediction.git
```

Move into the project folder:

```bash
cd Food-Delivery-Time-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶ Run the Project

Run the machine learning script:

```bash
python Food_Delivery_Prediction.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📷 Project Screenshots

### Dataset Preview

_Add your screenshot here_

### Exploratory Data Analysis

_Add EDA screenshots here_

### Model Evaluation

_Add model comparison screenshot here_

### Streamlit Application

_Add Streamlit screenshots here_

---

## 📌 Results

- Successfully preprocessed the dataset.
- Performed Exploratory Data Analysis.
- Trained Linear Regression and Decision Tree Regression models.
- Evaluated models using MAE, RMSE, and R² Score.
- Compared the performance of both models.
- Developed a Streamlit application for real-time delivery time prediction.

---

## 🔮 Future Improvements

- Random Forest Regression
- XGBoost Regression
- Live Traffic API Integration
- Live Weather API Integration
- Cloud Deployment
- Deep Learning-based Prediction

---

## 📚 References

- Scikit-learn Documentation
- Pandas Documentation
- Streamlit Documentation
- Kaggle Food Delivery Dataset
- Python Official Documentation

---

## 👨‍💻 Author

**Muhil Amuthan**

GitHub: https://github.com/muhil-amuthan

---

## ⭐ Support

If you found this project helpful, please consider giving it a ⭐ on GitHub.
