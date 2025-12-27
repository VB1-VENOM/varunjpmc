import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#Load census bureau data from files.
def load_census_data(data_path='census-bureau.data', columns_path='census-bureau.columns'):
    # Read column names
    with open(columns_path, 'r') as f:
        columns = [line.strip() for line in f.readlines() if line.strip()]
    
    # Read data
    df = pd.read_csv(data_path, header=None, names=columns)
    
    # Clean up the label column (remove trailing spaces and periods)
    df['label'] = df['label'].str.strip().str.rstrip('.')
    
    return df

#Preprocess data for classification task.
def preprocess_for_classification(df, test_size=0.2, random_state=42):
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Separate target variable
    y = (df_processed['label'] == '50000+').astype(int)
    
    # Remove target, weight, and year from features
    # Weight is for population representation, not a feature
    # Year is not useful for prediction
    X = df_processed.drop(['label', 'weight', 'year'], axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in X.columns:
        # Try to convert to numeric
        try:
            pd.to_numeric(X[col], errors='raise')
            numerical_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)
    
    # Handle missing values in numerical columns
    for col in numerical_cols:
        # Replace '?' and other non-numeric values with NaN
        X[col] = pd.to_numeric(X[col], errors='coerce')
        # Fill NaN with median
        X[col].fillna(X[col].median(), inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # Handle missing values in categorical columns
        X[col] = X[col].fillna('Unknown')
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    feature_names = list(X.columns)
    
    return X_train, X_test, y_train, y_test, feature_names, label_encoders, scaler

#Preprocess data for clustering/segmentation task.
def preprocess_for_clustering(df, n_samples=None, random_state=42):
    # Sample data if needed (for computational efficiency)
    df_sampled = df.copy()
    if n_samples and len(df_sampled) > n_samples:
        df_sampled = df_sampled.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
    
    # Create a copy
    df_processed = df_sampled.copy()
    
    # Remove weight and year, but keep label for analysis
    X = df_processed.drop(['weight', 'year'], axis=1)
    
    # Separate label for later analysis
    labels = X['label'].copy()
    X = X.drop('label', axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols = []
    categorical_cols = []
    
    for col in X.columns:
        try:
            pd.to_numeric(X[col], errors='raise')
            numerical_cols.append(col)
        except (ValueError, TypeError):
            categorical_cols.append(col)
    
    # Handle missing values in numerical columns
    for col in numerical_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col].fillna(X[col].median(), inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = X[col].fillna('Unknown')
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Scale all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    feature_names = list(X.columns)
    
    return X_scaled, feature_names, label_encoders, scaler, labels, df_sampled

