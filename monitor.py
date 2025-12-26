# =============================================================================
# Import Necessary Libraries
# =============================================================================

import os
import math
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn import set_config

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    f1_score, 
    recall_score,  
    classification_report, 
    PrecisionRecallDisplay, 
    ConfusionMatrixDisplay
)

from xgboost import XGBClassifier

import joblib

from evidently import Report
from evidently.presets import DataDriftPreset

warnings.filterwarnings(action='ignore') 
set_config(display='diagram')          

# =============================================================================
# Data Loading & Drift Simulation
# =============================================================================

def load_data(file_path='WA_Fn-UseC_-Telco-Customer-Churn.csv'):
    """
    Loads the dataset and simulates a 'current' dataset with artificial data drift.
    
    This function mimics a real-world scenario where the data distribution changes 
    over time (Concept Drift or Data Drift), necessitating model retraining.
    
    Args:
        file_path (str): The file path to the CSV dataset.
        
    Returns:
        ref_data (pd.DataFrame): The original baseline data (Reference).
        curr_data (pd.DataFrame): The simulated new data with noise and drift (Current).
    """
    # Load the reference data (baseline)
    ref_data = pd.read_csv(file_path)

    # Create a copy to serve as the 'current' incoming data
    curr_data = ref_data.copy()
    
    # Add noise and increase the magnitude of 'MonthlyCharges' to shift its mean/variance.
    noise = np.random.normal(0, 5, size=len(curr_data))
    curr_data['MonthlyCharges'] = (curr_data['MonthlyCharges'] * 1.2) + 10 + noise
    
    # Drastically change the distribution of the 'Contract' column.
    indices_to_change = curr_data.sample(frac=0.5, random_state=42).index
    curr_data.loc[indices_to_change, 'Contract'] = 'Month-to-month' 

    return ref_data, curr_data

# =============================================================================
# Drift Detection
# =============================================================================

def check_data_drift(ref_data, curr_data):
    """
    Detects data drift between the reference and current datasets using Evidently AI.
    
    It generates an HTML report visualizing the drift and returns a boolean 
    indicating if significant drift was found based on statistical tests.
    
    Args:
        ref_data (pd.DataFrame): The baseline data used for training the original model.
        curr_data (pd.DataFrame): The new batch of data to check against the baseline.
        
    Returns:
        bool: True if drift is detected in any feature, False otherwise.
    """
    # Initialize the drift report using the DataDriftPreset
    drift_report = Report(metrics=[
        DataDriftPreset()
    ])

    # Run the report calculation
    report = drift_report.run(reference_data=ref_data, current_data=curr_data)

    # Generate a timestamped filename for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"drift_report_{timestamp}.html"

    if not os.path.exists('reports/'):
        os.makedirs('reports/')
    save_path = os.path.join('reports/', filename)
    
    # Save the visual HTML report 
    report.save_html(save_path)
    print("Drift report saved to \"{}\".\n".format(save_path))

    # Parse the report dictionary to programmatically check for drifted features
    report_dict = report.dict()
    drifted_features = []

    for item in report_dict['metrics']:
        if 'column' in item['config']:
            column = item['config']['column']
            method = item['config']['method']
            score = item['value']
            threshold = item['config']['threshold']
            
            # If the drift score exceeds the threshold, record it
            if score > threshold:
                drifted_features.append({
                    'Column': column,
                    'Drift Score': round(score, 4), 
                    'Threshold': threshold,
                    'Method': method
                })

    # Display results
    df_result = pd.DataFrame(drifted_features)
    
    if not drifted_features:
        print("No drift detected in any feature.")
        return False # No retraining required
    else:
        print("Drift detected in the following features:")
        print(df_result)
        return True # Retraining required

# =============================================================================
# Feature Engineering & Preprocessing
# =============================================================================

def raw_preprocessor(df):
    """
    Performs initial raw data cleaning such as dropping IDs and fixing data types.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)    
        
    if 'SeniorCitizen' in df.columns:   
        df['SeniorCitizen'] = df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})

    return df


class ServiceValueSimplifier(BaseEstimator, TransformerMixin):
    """
    Custom Transformer to simplify redundant categorical values.
    
    Example: Merges 'No internet service' into 'No' for cleaner categories.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # List of columns that contain redundant 'No service' categories
        cols_to_fix = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'
        ]
        
        for col in cols_to_fix:
            if col in X.columns:
                X[col] = X[col].replace({'No internet service': 'No', 'No phone service': 'No'})
        return X

def get_pipeline():
    """
    Constructs the complete Machine Learning Pipeline (Preprocessing + Model).
    
    Returns:
        Pipeline: A Scikit-Learn Pipeline object containing the preprocessor and XGBoost classifier.
    """
    # Categorical Pipeline: Impute missing values -> One-Hot Encode
    cat_pipeline = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder(handle_unknown='ignore', drop='if_binary') 
    )

    # Numerical Pipeline: Impute missing values -> Standard Scale (Z-score)
    num_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler()
    )

    # Log-Transform Pipeline: For skewed features (e.g., TotalCharges)
    # Impute -> Log(x+1) -> Standard Scale
    log_pipeline = make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"), 
        StandardScaler()
    )

    # Feature Groups Definition
    cat_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
        'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    num_features = ['tenure', 'MonthlyCharges']
    log_features = ['TotalCharges']

    # Combine all preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer([
        ("cat", cat_pipeline, cat_features),
        ("num", num_pipeline, num_features),
        ("log", log_pipeline, log_features)
    ], remainder='drop')

    # Initialize the XGBoost Classifier
    xgb_clf = XGBClassifier(
            use_label_encoder=False, 
            eval_metric='aucpr', # Area Under the Precision-Recall Curve
            random_state=42,
            n_jobs=-1
        )
    
    # Assemble the final pipeline
    full_pipeline = Pipeline([
        ("raw_preprocessor", FunctionTransformer(raw_preprocessor)), # Step 1: Raw cleaning
        ("simplifier", ServiceValueSimplifier()),                    # Step 2: Custom simplification
        ("preprocessing", preprocessor),                             # Step 3: Encoding & Scaling
        ("XGBClassifier", xgb_clf)                                   # Step 4: Model Training
    ])
    
    return full_pipeline

# =============================================================================
# Model Retraining & Evaluation
# =============================================================================

def retrain_model(ref_data, curr_data):
    """
    Retrains the model on the combined dataset, performs hyperparameter tuning,
    and saves the new model if it meets the performance criteria.
    
    Args:
        ref_data (pd.DataFrame): Old data.
        curr_data (pd.DataFrame): New data.
        
    Returns:
        estimator: The best trained model if successful, otherwise None.
    """
    # Merge datasets to train on the most recent information
    combined_data = pd.concat([ref_data, curr_data], axis=0).reset_index(drop=True)
    
    # Encode the target variable 'Churn' (No=0, Yes=1)
    if 'Churn' in combined_data.columns:
        combined_data['Churn'] = combined_data['Churn'].replace({'No': 0, 'Yes': 1})
        
    # Split into Train and Test sets with stratification
    train_data, test_data = train_test_split(combined_data, test_size=0.2, stratify=combined_data['Churn'], random_state=42)
    train_X = train_data.drop('Churn', axis=1)
    train_y = train_data['Churn'].copy()

    test_X = test_data.drop('Churn', axis=1)
    test_y = test_data['Churn'].copy()

    # Load the untrained pipeline structure
    pipeline = get_pipeline() 

    # Define Hyperparameter Search Grid
    param_grid = [
        {
        'XGBClassifier__n_estimators': [100, 300, 500, 700, 900],
        'XGBClassifier__max_depth': [3, 5, 7, 9],              
        'XGBClassifier__learning_rate': [0.01, 0.1, 0.2],   
        'XGBClassifier__subsample': [0.6, 0.8, 1.0],              
        'XGBClassifier__scale_pos_weight': [1, 2, 3]        
        }
    ]

    # Initialize Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1',  
        cv=10,          
        n_jobs=-1,     
        verbose=1
    )

    print("Starting Grid Search for Hyperparameter Tuning...")
    grid_search.fit(train_X, train_y)
    best_model = grid_search.best_estimator_

    # Evaluate the best model on the test set
    pred_y = best_model.predict(test_X)
    test_recall = recall_score(test_y, pred_y, pos_label=1)
    test_f1 = f1_score(test_y, pred_y, pos_label=1)
    
    # Define Acceptance Criteria (Thresholds)
    min_recall = 0.7
    min_f1 = 0.6

    print(f"New Model Performance - Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    # Decision Gate: Only save if the model is good enough
    if (test_recall >= min_recall) and (test_f1 >= min_f1):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"churn_model_{timestamp}.pkl"

        if not os.path.exists('models/'):
            os.makedirs('models/')
        save_path = os.path.join('models/', filename)
        
        # Serialize and save the model
        joblib.dump(best_model, save_path)
        print(f"Model successfully saved to: {save_path}")

        return best_model
    
    else:
        print("The retrained model did not meet the performance criteria. Retraining aborted.")
        if test_recall < min_recall:
            print(f"Fail: Recall {test_recall:.4f} < {min_recall}")
        if test_f1 < min_f1:
            print(f"Fail: F1 Score {test_f1:.4f} < {min_f1}")
        return None

# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # Check if the dataset exists locally
    file_name = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    if os.path.exists(file_name):
        print("Loading and preparing data...")
        ref_data, curr_data = load_data(file_name)
        
        print("Checking for Data Drift...")
        needs_retraining = check_data_drift(ref_data, curr_data)
        
        if needs_retraining:
            print("Drift detected! Initiating automated retraining process...")
            retrain_model(ref_data, curr_data)
        else:
            print("Data distribution is stable. No retraining necessary.")
    else:
        print(f"Error: Dataset '{file_name}' not found. Please ensure the file is in the current directory.")