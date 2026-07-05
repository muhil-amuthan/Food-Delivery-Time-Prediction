# 🍕 Food Delivery Time Prediction Using Machine Learning

A Machine Learning project that predicts food delivery time based on factors such as distance, weather, traffic conditions, preparation time, courier experience, and vehicle type. The project compares **Linear Regression** and **Decision Tree Regression**, and includes a **Streamlit web application** for real-time predictions.

## 🚀 Live Demo

🌐 **Streamlit App:** https://food-delivery-time-prediction-0728.streamlit.app/

---

## 📌 Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Linear Regression model
- Decision Tree Regression model
- Model performance comparison
- Real-time prediction using Streamlit
- User-friendly web interface

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

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

- Order_ID
- Distance_km
- Weather
- Traffic_Level
- Time_of_Day
- Vehicle_Type
- Preparation_Time_min
- Courier_Experience_yrs
- Delivery_Time_min (Target Variable)

---

## 🤖 Machine Learning Models

### Linear Regression
Predicts delivery time using a linear relationship between input features and the target variable.

### Decision Tree Regression
Learns complex relationships between features to improve prediction accuracy.

---

## 📈 Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

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

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python Food_Delivery_Prediction.py
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📷 Screenshots

### Home Page

_Add a screenshot here_

### Prediction Result

_Add a screenshot here_

### Model Comparison

_Add a screenshot here_

---

## 🌐 Live Application

👉 https://food-delivery-time-prediction-0728.streamlit.app/

---

## 📌 Future Improvements

- Random Forest Regression
- XGBoost Regression
- Live Traffic API Integration
- Weather API Integration
- Cloud Deployment
- Mobile Application

---

## 👨‍💻 Author

**Muhil Amuthan**

GitHub: https://github.com/muhil-amuthan

---

⭐ If you found this project useful, please consider starring the repository.
