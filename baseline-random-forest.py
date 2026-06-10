# 🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Step 2: Load Dataset
df = pd.read_csv("StudentPerformanceFactors.csv")

# 🔹 Step 3: Basic Cleaning (optional but safe)
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)

# 🔹 Step 4: Select Features & Target
# 👉 Change 'Exam_Score' to your actual target column if different
target_column = 'Exam_Score'

X = df.drop(target_column, axis=1)
y = df[target_column]

# 🔹 Step 5: Handle Categorical Data (IMPORTANT)
X = pd.get_dummies(X, drop_first=True)

# 🔹 Step 6: Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🔹 Step 7: Train Model (Random Forest 🔥)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# 🔹 Step 8: Predictions
y_pred = model.predict(X_test)

# 🔹 Step 9: Evaluation
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("R2 Score:", r2)

# 🔹 Step 10: Visualization (IMPORTANT FOR VIVA)
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()