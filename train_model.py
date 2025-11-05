import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

print("Loading dataset...")
DATA_FILE = 'solar_data.csv'
MODEL_FILE = 'solar_model.joblib'

try:
    data = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Error: '{DATA_FILE}' not found.")
    print("Please run 'python create_dataset.py' first.")
    exit()

print(f"Dataset loaded with {len(data)} rows.")

# 1. Define Features (X) and Target (y)
FEATURES = ['hour_of_day', 'month', 'temperature_c', 'radiation_w_m2']
TARGET = 'kwh_generated'

X = data[FEATURES]
y = data[TARGET]

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training model on {len(X_train)} samples...")

# 3. Initialize and Train the Model
# RandomForest is an "ensemble" model, meaning it combines many
# small "decision trees" to make a more accurate and robust prediction.
# 
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15, oob_score=True)
model.fit(X_train, y_train)

print("Model training complete.")

# 4. Evaluate the Model (Optional, but good practice)
print("Evaluating model performance...")
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2 = r2_score(y_test, y_pred)
oob = model.oob_score_

print(f"  Model RMSE (Root Mean Squared Error): {rmse:.4f} kWh")
print(f"  Model R^2 Score (Coefficient of Determination): {r2:.4f}")
print(f"  Model OOB Score (Out-of-Bag): {oob:.4f}")

# 5. Save the trained model to a file
joblib.dump(model, MODEL_FILE)

print(f"\nSuccessfully trained and saved model to '{MODEL_FILE}'")
print(f"File saved to: {os.path.abspath(MODEL_FILE)}")
print("\nYou can now run the app with: streamlit run app.py")
