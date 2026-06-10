# Student Performance Prediction

## Overview

Student Performance Prediction is an end-to-end Machine Learning project designed to analyze and predict student academic performance using lifestyle, academic, and demographic factors.

The project focuses on data preprocessing, exploratory data analysis, feature engineering, and predictive modeling to estimate student exam scores. By analyzing factors such as study habits, attendance, sleep patterns, previous academic performance, and access to resources, the system identifies the key contributors to student success.

The project implements Random Forest Regression models and performs feature importance analysis to understand which factors have the greatest impact on academic outcomes.

---

## Objectives

* Analyze factors affecting student academic performance
* Build a robust data preprocessing pipeline
* Perform feature engineering to improve prediction quality
* Train and evaluate machine learning models
* Identify the most influential factors affecting exam scores

---

## Dataset

The project uses the Student Performance Factors dataset containing academic, lifestyle, and demographic information about students.

### Key Features

* Hours Studied
* Attendance
* Sleep Hours
* Previous Scores
* Physical Activity
* Tutoring Sessions
* Teacher Quality
* Access to Resources
* Motivation Level
* Family Income
* Internet Access
* Parental Education Level
* School Type
* Gender

### Target Variable

* Exam Score

---

## Project Workflow

### 1. Data Preprocessing

* Missing value treatment
* Duplicate removal
* Outlier detection using IQR
* Outlier removal
* Categorical feature encoding
* Feature scaling using StandardScaler

### 2. Exploratory Data Analysis

* Missing value visualization
* Correlation heatmaps
* Distribution analysis
* Regression plots
* Feature relationship analysis

### 3. Feature Engineering

Custom features were created to improve predictive performance:

* Study Efficiency
* Consistency Score
* Sleep Quality Indicator
* Stress Level
* Lifestyle Score

### 4. Machine Learning Model

**Algorithm Used**

* Random Forest Regression

### Model Capabilities

* Predict student exam scores
* Analyze feature importance
* Capture non-linear relationships
* Handle mixed feature types effectively

---

## Visualizations

The project generates:

* Missing Value Heatmaps
* Outlier Detection Boxplots
* Correlation Heatmaps
* Feature Importance Charts
* Actual vs Predicted Graphs
* Distribution Analysis
* Regression Trend Plots

---

## Tech Stack

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

## Machine Learning Pipeline

```text
Raw Dataset
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Outlier Removal
      ↓
Feature Engineering
      ↓
Feature Scaling
      ↓
Train-Test Split
      ↓
Random Forest Regression
      ↓
Performance Evaluation
      ↓
Feature Importance Analysis
```

---

## Results

The model successfully predicts student academic performance using lifestyle and academic indicators.

### Key Findings

* Study habits significantly influence performance
* Attendance strongly correlates with exam scores
* Previous academic performance is one of the strongest predictors
* Sleep and lifestyle factors impact academic outcomes
* Feature engineering improved model interpretability and analysis

---

## Repository Structure

```text
student-performance-prediction/
│
├── StudentPerformanceFactors.csv
├── data-cleaning-pipeline.py
├── feature-engineering-random-forest.py
├── baseline-random-forest.py
├── screenshots/
│   ├── correlation-heatmap.png
│   ├── feature-importance.png
│   ├── actual-vs-predicted.png
│   ├── missing-values.png
│   └── regression-analysis.png
├── README.md
└── LICENSE
```

---

## Author

**Tanush Bhalla**

B.Tech Computer Science Engineering (AI/ML)

Passionate about Machine Learning, Data Analytics, Mobile Development, and Building Real-World Software Solutions.
