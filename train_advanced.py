import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib

print("Starting training script...")

# 1. Load Data
df = pd.read_csv("employee_data.csv")

# 2. Feature Engineering
# Convert 'DateOfJoining' to datetime objects
df['DateOfJoining'] = pd.to_datetime(df['DateOfJoining'])
# Calculate 'YearsExperience'
df['YearsExperience'] = (datetime.now() - df['DateOfJoining']).dt.days / 365.25
print("Feature 'YearsExperience' created.")

# 3. Define Features (X) and Target (y)
X = df[['YearsExperience', 'EducationLevel']]
y = df['Salary']

# 4. Define Preprocessing
# We have one numerical feature and one categorical feature.
# They need to be treated differently.

# Define the order for EducationLevel
education_levels = ["High School", "Bachelor's", "Master's", "PhD"]

# Create transformers
# 'passthrough' means 'YearsExperience' will be scaled
numeric_transformer = StandardScaler()

# 'OrdinalEncoder' will convert text to numbers (e.g., High School=0, Bachelor's=1)
categorical_transformer = OrdinalEncoder(categories=[education_levels], handle_unknown='use_encoded_value', unknown_value=-1)

# 5. Create a ColumnTransformer
# This applies the correct transformer to the correct column
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, ['YearsExperience']),
        ('cat', categorical_transformer, ['EducationLevel'])
    ])

# 6. Create the Full ML Pipeline
# This pipeline will first run the 'preprocessor' and then run the 'regressor'
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 7. Split, Train, and Evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)
print("Model pipeline trained.")

# 8. Evaluate to get the RMSE (error) for our range
y_pred = model_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model R-squared: {model_pipeline.score(X_test, y_test):.2f}")
print(f"Model RMSE: ${rmse:,.2f}")

# 9. Save the Pipeline AND the RMSE value
joblib.dump(model_pipeline, 'salary_pipeline.pkl')
joblib.dump(rmse, 'model_rmse.pkl')

print("Pipeline and RMSE saved successfully!")