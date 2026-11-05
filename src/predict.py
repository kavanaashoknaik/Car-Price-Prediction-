import joblib
import pandas as pd

# Load model
model = joblib.load("car_price_model.pkl")

print("=== Car Price Prediction ===")

# User Inputs
year = int(input("Enter Car Year: "))
present_price = float(input("Enter Current Showroom Price (in lakhs): "))
kms_driven = int(input("Enter Kilometers Driven: "))
fuel_type = input("Enter Fuel Type (Petrol/Diesel): ")
transmission = input("Enter Transmission (Manual/Automatic): ")
owner = int(input("Enter Number of Previous Owners: "))

# Convert text values into numbers
fuel_type_value = 0 if fuel_type.lower() == "petrol" else 1
transmission_value = 0 if transmission.lower() == "manual" else 1

# Create dataframe
new_car = pd.DataFrame({
    "Year": [year],
    "Present_Price": [present_price],
    "Kms_Driven": [kms_driven],
    "Fuel_Type": [fuel_type_value],
    "Transmission": [transmission_value],
    "Owner": [owner]
})

# Predict
prediction = model.predict(new_car)

print("\\nEstimated Car Selling Price:")
print(f"₹ {prediction[0]:.2f} Lakhs")