# Food_Delivery_Prediction.py
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

# Fill missing values
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

encoders = {}
for col in df.select_dtypes(exclude="number").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("\nEncoded category mappings:")
for col, le in encoders.items():
    print(f"  {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

print("\nCorrelation:")
print(df.corr(numeric_only=True))

plt.figure(figsize=(8, 5))
df["Delivery_Time_min"].hist(bins=20)
plt.title("Delivery Time Distribution")
plt.xlabel("Minutes")
plt.ylabel("Count")
plt.show()

for c in ["Weather", "Traffic_Level", "Vehicle_Type", "Time_of_Day"]:
    if c in df.columns:
        plt.figure(figsize=(6, 4))
        df.groupby(c)["Delivery_Time_min"].mean().plot(kind="bar")
        plt.title(f"Average Delivery Time by {c}")
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
print("\n💾 Saved: delivery_model.pkl")

joblib.dump(encoders, "encoders.pkl")
print("💾 Saved: encoders.pkl")