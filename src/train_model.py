import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("../dataset/car_data.csv")

print("Dataset Loaded Successfully")
print(df.head())


# Convert categorical values into numbers
df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["Transmission"] = df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

# Features
X = df[[
    "Year",
    "Present_Price",
    "Kms_Driven",
    "Fuel_Type",
    "Transmission",
    "Owner"
]]

# Target
y = df["Selling_Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

print("Model Training Completed")

# Predict
predictions = model.predict(X_test)

# Check error
error = mean_absolute_error(y_test, predictions)

print(f"Mean Absolute Error: {error}")

# Save model
joblib.dump(model, "car_price_model.pkl")

print("Model Saved Successfully")