import pandas as pd
import json
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



DATA_PATH = "dataset/winequality.csv"
data = pd.read_csv(DATA_PATH, sep=';')


target_column = "quality"

X = data.drop(columns=[target_column])
y = data[target_column]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train_scaled, y_train)


y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


joblib.dump(model, "output/model.pkl")

metrics = {
    "mean_squared_error": mse,
    "r2_score": r2
}

with open("output/results.json", "w") as f:
    json.dump(metrics, f, indent=4)


print("Model Evaluation Metrics")
print("------------------------")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")
