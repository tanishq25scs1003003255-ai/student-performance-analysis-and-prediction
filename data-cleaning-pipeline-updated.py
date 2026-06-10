import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("screenshots", exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (10,5)

main_color = "#2563eb"
accent_color = "#f59e0b"
success_color = "#10b981"

# LOAD DATA

df = pd.read_csv("StudentPerformanceFactors.csv")
print("Initial Shape:", df.shape)

# MISSING VALUES HEATMAP (BEFORE)

plt.figure()
sns.heatmap(df.isnull(), cbar=False, cmap="Blues")
plt.title("Missing Values Overview", weight='bold')
plt.tight_layout()
plt.savefig("screenshots/missing-values-before-cleaning.png", dpi=300, bbox_inches="tight")
plt.show()

# REMOVE DUPLICATES

df.drop_duplicates(inplace=True)

# HANDLE MISSING VALUES

df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0], inplace=True)
df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0], inplace=True)
df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0], inplace=True)

# AFTER CLEANING

plt.figure()
sns.heatmap(df.isnull(), cbar=False, cmap="Greens")
plt.title("After Missing Value Treatment", weight='bold')
plt.tight_layout()
plt.savefig("screenshots/missing-values-after-cleaning.png", dpi=300, bbox_inches="tight")
plt.show()

# OUTLIERS BEFORE

num_cols = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours',
    'Previous_Scores', 'Tutoring_Sessions',
    'Physical_Activity', 'Exam_Score'
]

fig, axes = plt.subplots(2,2, figsize=(12,8))
axes = axes.flatten()

for i, col in enumerate(num_cols[:4]):
    sns.boxplot(y=df[col], ax=axes[i], color=main_color)
    axes[i].set_title(col, weight='bold')

plt.suptitle("Outliers Before Cleaning", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("screenshots/outliers-before-cleaning.png", dpi=300, bbox_inches="tight")
plt.show()

# REMOVE OUTLIERS (IQR)

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df = df[(df[col] >= lower) & (df[col] <= upper)]

# OUTLIERS AFTER

fig, axes = plt.subplots(2,2, figsize=(12,8))
axes = axes.flatten()

for i, col in enumerate(num_cols[:4]):
    sns.boxplot(y=df[col], ax=axes[i], color=success_color)
    axes[i].set_title(f"{col} (Cleaned)", weight='bold')

plt.suptitle("Outliers After Cleaning", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("screenshots/outliers-after-cleaning.png", dpi=300, bbox_inches="tight")
plt.show()

# ENCODING

cat_cols = [
    'Parental_Involvement', 'Access_to_Resources',
    'Extracurricular_Activities', 'Motivation_Level',
    'Internet_Access', 'Family_Income',
    'Teacher_Quality', 'School_Type',
    'Peer_Influence', 'Learning_Disabilities',
    'Parental_Education_Level', 'Distance_from_Home',
    'Gender'
]

df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# FEATURE SCALING

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# FINAL DISTRIBUTION

plt.figure()
sns.histplot(df['Exam_Score'], bins=30, kde=True,
             color=main_color, edgecolor='black')

plt.axvline(df['Exam_Score'].mean(), color=accent_color,
            linestyle='--', label='Mean')

plt.title("Final Exam Score Distribution", weight='bold')
plt.legend()
plt.tight_layout()
plt.savefig("screenshots/exam-score-distribution.png", dpi=300, bbox_inches="tight")
plt.show()

# CORRELATION HEATMAP

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)

plt.title("Feature Correlation Heatmap", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("screenshots/correlation-heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# REGRESSION PATTERN GRAPHS

top_features = ['Hours_Studied', 'Attendance', 'Previous_Scores']

fig, axes = plt.subplots(1,3, figsize=(18,5))

for i, col in enumerate(top_features):
    sns.regplot(x=df[col], y=df['Exam_Score'],
                ax=axes[i],
                scatter_kws={'alpha':0.5},
                line_kws={'color':'red'})
    
    axes[i].set_title(f"{col} vs Exam Score", weight='bold')

plt.suptitle("Key Factors Affecting Exam Score", fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig("screenshots/regression-analysis.png", dpi=300, bbox_inches="tight")
plt.show()

# FEATURE IMPORTANCE STYLE GRAPH

corr_target = df.corr()['Exam_Score'].sort_values(ascending=False)[1:10]

plt.figure(figsize=(8,5))
sns.barplot(x=corr_target.values, y=corr_target.index, palette="Blues_r")

plt.title("Top Features Influencing Exam Score", weight='bold')
plt.xlabel("Correlation Strength")
plt.tight_layout()
plt.savefig("screenshots/feature-importance.png", dpi=300, bbox_inches="tight")
plt.show()

# PAIRPLOT

pairplot = sns.pairplot(
    df[['Hours_Studied','Attendance','Previous_Scores','Exam_Score']],
    kind='reg',
    plot_kws={'line_kws':{'color':'red'}}
)

pairplot.fig.suptitle("Multi-variable Relationships", y=1.02)
pairplot.savefig("screenshots/multi-variable-relationships.png", dpi=300)

plt.show()

df.to_csv("cleaned_student_data.csv", index=False)

print("Final Shape:", df.shape)
print("Pipeline Completed")
