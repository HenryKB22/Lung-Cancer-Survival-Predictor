# Lung Cancer Survival Prediction Project - Documentation

## Overview

This Jupyter Notebook presents a comprehensive workflow for predicting lung cancer survival using a real-world dataset. The notebook covers data loading, preprocessing, feature engineering, exploratory data analysis (EDA), handling class imbalance, model training, and evaluation using various machine learning algorithms and resampling techniques.

---

## Sections

### 1. Data Loading and Preprocessing

- **Libraries Imported:** numpy, pandas, matplotlib, seaborn, scikit-learn, imbalanced-learn, xgboost.
- **Dataset:** Loaded from `cancer_dataset.csv` into a pandas DataFrame (`df`).
- **Initial Exploration:** Used `.head()`, `.info()`, `.shape`, `.isnull().sum()`, and `.describe()` to understand the data structure and check for missing values.
- **Date Conversion:** Converted `diagnosis_date` and `end_treatment_date` columns to datetime objects.

### 2. Feature Engineering

- **Function `engineer_features(df)`:** 
    - Adds new features:
        - `treatment_duration_days`: Days between diagnosis and end of treatment.
        - `age_group`: Categorical age bins.
        - `bmi_category`: BMI categories based on WHO standards.
- **Application:** Engineered features are added to the main DataFrame.

### 3. Exploratory Data Analysis (EDA)

- **Visualizations:** 
    - Distribution plots for BMI category, age group, survival, age, gender, and cancer stage.
    - Correlation heatmap for numeric features.
- **Purpose:** To understand feature distributions, class imbalance, and relationships between variables.

### 4. Model Selection & Training

- **Feature Selection:** 
    - Dropped non-predictive columns and target variable (`survived`).
    - Applied one-hot encoding to categorical variables.
- **Train-Test Split:** 
    - Used `train_test_split` with stratification to maintain class distribution.
- **Scaling & Dimensionality Reduction:** 
    - Used `StandardScaler` and `PCA` in a pipeline for logistic regression.

### 5. Handling Class Imbalance

- **Undersampling:** 
    - `EditedNearestNeighbours` and `RandomUnderSampler` to balance classes by removing samples from the majority class.
- **Oversampling:** 
    - `SMOTE`, `ADASYN`, and `RandomOverSampler` to synthetically generate samples for the minority class.
- **Resampled Datasets:** 
    - Created new balanced datasets for training.

### 6. Model Training and Evaluation

- **Models Used:** 
    - Logistic Regression, Random Forest, XGBoost.
- **Metrics:** 
    - Accuracy, F1 Score, Confusion Matrix, and Classification Report.
- **Evaluation:** 
    - Compared model performance on original and resampled datasets.
    - Visualized confusion matrices for each approach.

---

## Key Variables

- **df:** Main DataFrame containing the dataset and engineered features.
- **X, y:** Features and target variable for modeling.
- **X_train, X_test, y_train, y_test:** Train-test splits.
- **X_resampled, y_resampled:** Resampled datasets for handling class imbalance.
- **log_reg_model, rf_model, xgb_model:** Machine learning models.
- **scaler:** StandardScaler instance for feature scaling.
- **Various predictions:** e.g., `y_pred_enn`, `y_pred_under`, `y_pred_smote`, etc.

---

## Conclusion

This notebook demonstrates a robust machine learning pipeline for survival prediction, including:
- Data cleaning and feature engineering.
- EDA for insights and class imbalance detection.
- Multiple strategies for handling imbalanced classes.
- Training and evaluation of several classifiers.
- Visualization of results for interpretability.

The workflow can be adapted for similar medical prediction tasks or extended with further feature engineering and hyperparameter tuning.