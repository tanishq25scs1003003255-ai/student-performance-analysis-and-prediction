# 🔹 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Step 2: Load Dataset
df = pd.read_csv("cleaned_student_data.csv")

# 🔹 Step 3: Basic Cleaning
df.drop_duplicates(inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)


df['study_efficiency'] = df['Hours_Studied'] / (df['Sleep_Hours'] + 1)
df['consistency'] = df['Hours_Studied'] * df['Attendance']
df['sleep_quality'] = df['Sleep_Hours'].apply(lambda x: 1 if x >= 7 else 0)
df['stress_level'] = df['Hours_Studied'] / (df['Sleep_Hours'] + 1)
df['lifestyle_score'] = df['Sleep_Hours'] + df['Attendance'] - df['Hours_Studied']


print(df[['study_efficiency', 'consistency', 'sleep_quality',
          'stress_level', 'lifestyle_score']].head())

#VISUALIZATION

# Histogram of Study Efficiency
plt.figure()
df['study_efficiency'].hist()
plt.title("Study Efficiency Distribution")
plt.xlabel("Study Efficiency")
plt.ylabel("Frequency")
plt.show()

# Boxplot for Consistency
plt.figure()
sns.boxplot(x=df['consistency'])
plt.title("Consistency (Outlier Detection)")
plt.show()

# Bar plot for Sleep Quality
plt.figure()
sns.countplot(x=df['sleep_quality'])
plt.title("Sleep Quality Distribution")
plt.show()

# Scatter plot: Lifestyle vs Exam Score
plt.figure()
plt.scatter(df['lifestyle_score'], df['Exam_Score'])
plt.xlabel("Lifestyle Score")
plt.ylabel("Exam Score")
plt.title("Lifestyle vs Performance")
plt.show()

# MODEL PREPARATION
target_column = 'Exam_Score'

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert categorical
X = pd.get_dummies(X, drop_first=True)

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model (Random Forest)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

#  Predictions
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import mean_squared_error, r2_score

print("\n Model Performance:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Actual vs Predicted Graph
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.show()

# FEATURE IMPORTANCE
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nsome attributes that matter the most\n")
print(importance_df.head(10))

# Plot feature importance
print("\nbar graph for important attributes that contribute the maximum\n")
plt.figure()
importance_df.head(10).plot(kind='bar', x='Feature', y='Importance')
plt.title("Top Features Affecting Performance")
plt.xticks(rotation=45)
plt.show()