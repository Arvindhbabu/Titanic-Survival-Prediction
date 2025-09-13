# Titanic Survival Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Overview

This project implements a machine learning pipeline to predict passenger survival on the RMS Titanic using the classic Titanic dataset. The model leverages a Random Forest Classifier to analyze features such as passenger class, sex, age, fare, and embarkation port, achieving an accuracy of 82% on the test set. It demonstrates essential data science workflows, including data cleaning, feature encoding, scaling, and model evaluation.

The project is built with Python and scikit-learn, suitable for educational purposes or as a baseline for advanced ensemble methods.

## Problem Statement

The RMS Titanic sank on April 15, 1912, after colliding with an iceberg, resulting in over 1,500 fatalities out of 2,224 passengers and crew. Survival was influenced by factors like socioeconomic status, gender, and family ties, rather than pure chance.

This analysis builds a predictive model to classify passengers as survivors or non-survivors based on their features, uncovering insights into disparity in lifeboat access and evacuation priorities.

## Dataset

- **Source**: Kaggle Titanic Dataset [](https://www.kaggle.com/c/titanic/data)
- **Size**: 891 training samples, 418 test samples (this project uses the combined training set for simplicity)
- **Features**:
  - `Survived`: Target variable (0 = No, 1 = Yes)
  - `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)
  - `Sex`: Gender (male/female)
  - `Age`: Age in years
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Fare`: Passenger fare
  - `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **Preprocessing Notes**: Dropped irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`); imputed missing `Age` with median and `Embarked` with mode; encoded categorical variables.

## Methodology

1. **Data Loading and Exploration**: Load CSV data and inspect structure using Pandas.
2. **Preprocessing**:
   - Handle missing values.
   - Encode `Sex` and `Embarked` using LabelEncoder.
   - Drop non-predictive columns.
3. **Feature Preparation**:
   - Split into training (80%) and testing (20%) sets.
   - Scale numerical features with StandardScaler.
4. **Model Training**: Fit a Random Forest Classifier (100 estimators, random_state=42).
5. **Evaluation**: Compute accuracy and classification report (precision, recall, F1-score).

The pipeline emphasizes robustness to missing data and categorical encoding for tree-based models.

## Results

- **Model Accuracy**: 82% on the test set.
- **Classification Report** (Test Set):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Non-Survivor) | 0.83 | 0.88 | 0.85 | 105 |
| 1 (Survivor) | 0.81 | 0.74 | 0.77 | 74 |
| **Macro Avg** | 0.82 | 0.81 | 0.81 | 179 |
| **Weighted Avg** | 0.82 | 0.82 | 0.82 | 179 |

Key Insights:
- Females and higher-class passengers had higher survival rates.
- The model performs well on non-survivors but slightly underperforms on survivors, indicating potential for hyperparameter tuning.
