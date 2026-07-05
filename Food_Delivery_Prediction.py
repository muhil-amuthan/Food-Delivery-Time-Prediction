import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Food_Delivery_Times.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nInfo:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


df.drop_duplicates(inplace=True)

df.drop(columns=["Order_ID"], inplace=True)

df["Distance_km"] = df["Distance_km"].fillna(df["Distance_km"].median())
df["Delivery_Time_min"] = df["Delivery_Time_min"].fillna(df["Delivery_Time_min"].median())
df["Preparation_Time_min"] = df["Preparation_Time_min"].fillna(df["Preparation_Time_min"].median())

# Fill missing text with the most common value (mode)
df["Weather"] = df["Weather"].fillna(df["Weather"].mode()[0])
df["Traffic_Level"] = df["Traffic_Level"].fillna(df["Traffic_Level"].mode()[0])
df["Vehicle_Type"] = df["Vehicle_Type"].fillna(df["Vehicle_Type"].mode()[0])
df["Time_of_Day"] = df["Time_of_Day"].fillna(df["Time_of_Day"].mode()[0])


encoders = {}
le_weather = LabelEncoder()
df["Weather"] = le_weather.fit_transform(df["Weather"])
encoders["Weather"] = le_weather

le_traffic = LabelEncoder()
df["Traffic_Level"] = le_traffic.fit_transform(df["Traffic_Level"])
encoders["Traffic_Level"] = le_traffic

le_vehicle = LabelEncoder()
df["Vehicle_Type"] = le_vehicle.fit_transform(df["Vehicle_Type"])
encoders["Vehicle_Type"] = le_vehicle

le_time = LabelEncoder()
df["Time_of_Day"] = le_time.fit_transform(df["Time_of_Day"])
encoders["Time_of_Day"] = le_time

print("\nEncoded category mappings:")
print("  Weather:", dict(zip(le_weather.classes_, le_weather.transform(le_weather.classes_))))
print("  Traffic_Level:", dict(zip(le_traffic.classes_, le_traffic.transform(le_traffic.classes_))))
print("  Vehicle_Type:", dict(zip(le_vehicle.classes_, le_vehicle.transform(le_vehicle.classes_))))
print("  Time_of_Day:", dict(zip(le_time.classes_, le_time.transform(le_time.classes_))))

print("\nCorrelation:")
print(df.corr(numeric_only=True))

plt.figure(figsize=(8, 5))
df["Delivery_Time_min"].hist(bins=20)
plt.title("Delivery Time Distribution")
plt.xlabel("Minutes")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
df.groupby("Weather")["Delivery_Time_min"].mean().plot(kind="bar")
plt.title("Average Delivery Time by Weather")
plt.ylabel("Minutes")
plt.show()

plt.figure(figsize=(6, 4))
df.groupby("Traffic_Level")["Delivery_Time_min"].mean().plot(kind="bar")
plt.title("Average Delivery Time by Traffic Level")
plt.ylabel("Minutes")
plt.show()

plt.figure(figsize=(6, 4))
df.groupby("Vehicle_Type")["Delivery_Time_min"].mean().plot(kind="bar")
plt.title("Average Delivery Time by Vehicle Type")
plt.ylabel("Minutes")
plt.show()

plt.figure(figsize=(6, 4))
df.groupby("Time_of_Day")["Delivery_Time_min"].mean().plot(kind="bar")
plt.title("Average Delivery Time by Time of Day")
plt.ylabel("Minutes")
plt.show()

X = df.drop("Delivery_Time_min", axis=1)
y = df["Delivery_Time_min"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nFeature columns (X):")
print(list(X.columns))

print("\n===== Linear Regression =====")
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

print("MAE :", lr_mae)
print("RMSE:", lr_rmse)
print("R2  :", lr_r2)

print("\n===== Decision Tree =====")
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)

print("MAE :", dt_mae)
print("RMSE:", dt_rmse)
print("R2  :", dt_r2)

comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree"],
    "MAE": [lr_mae, dt_mae],
    "RMSE": [lr_rmse, dt_rmse],
    "R2": [lr_r2, dt_r2]
})

print("\nModel Comparison")
print(comparison)

best_name = comparison.loc[comparison["R2"].idxmax(), "Model"]
best_model = lr if best_name == "Linear Regression" else dt
print("\nBest Model:", best_name)

joblib.dump(best_model, "delivery_model.pkl")
print("\nSaved: delivery_model.pkl")

joblib.dump(encoders, "encoders.pkl")
print("Saved: encoders.pkl")
