import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import mean_squared_error, r2_score # <-- IMPORT THESE
import numpy as np # <-- IMPORT THIS

# 1. Load the dataset
data = pd.read_csv("Salary_Data.csv")

# 2. Define features (X) and target (y)
X = data[['YearsExperience']] 
y = data['Salary']

# 3. Split the data into training and testing sets
# We use 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create and Train the model
model = LinearRegression()
model.fit(X_train, y_train) # Train the model on the training data

# 5. Evaluate the Model (on the test set)
print("Evaluating model...")
y_pred = model.predict(X_test) # Predict using the test data

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- MODEL EVALUATION RESULTS ---")
print(f"R-squared (R²) score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print("---------------------------------")
print("This R-squared score is very high because the dummy data is very clean.")


# 6. Save the trained model
# (We still train on the FULL dataset for the final app)
model_final = LinearRegression()
model_final.fit(X, y)
joblib.dump(model_final, "salary_model.pkl")

print("\nModel saved as salary_model.pkl")