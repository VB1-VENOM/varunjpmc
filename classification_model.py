import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import warnings
warnings.filterwarnings('ignore')

from load_data import load_census_data, preprocess_for_classification

#Train and evaluate a classification model.
def train_classification_model(X_train, X_test, y_train, y_test, model_type='random_forest'):
    print(f"\n{'='*60}")
    print(f"Training {model_type} model...")
    print(f"{'='*60}\n")
    
    # Initialize model
    if model_type == 'random_forest':
        # Use class_weight to handle imbalanced data
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Model Performance: {model_type.upper()}")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted <=$50k  Predicted >$50k")
    print(f"Actual <=$50k        {cm[0,0]:8d}        {cm[0,1]:8d}")
    print(f"Actual >$50k          {cm[1,0]:8d}        {cm[1,1]:8d}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['<=$50k', '>$50k']))
    
    return model, results

#Get feature importance from tree-based models.
def get_feature_importance(model, feature_names, top_n=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        importance_data = {
            'feature': [feature_names[i] for i in indices],
            'importance': importances[indices]
        }
        importance_df = pd.DataFrame(importance_data)
        return importance_df
    else:
        return None


def main():
    """Main function to run classification model training and evaluation."""
    print("="*60)
    print("CENSUS BUREAU INCOME CLASSIFICATION MODEL")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_census_data()
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"Class distribution:")
    print(df['label'].value_counts())
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, label_encoders, scaler = \
        preprocess_for_classification(df)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Number of features: {len(feature_names)}")
    
    # Train multiple models
    models = {}
    results = {}
    
    # Random Forest
    rf_model, rf_results = train_classification_model(
        X_train, X_test, y_train, y_test, model_type='random_forest'
    )
    models['random_forest'] = rf_model
    results['random_forest'] = rf_results
    
    # Get feature importance
    print("\n" + "="*60)
    print("TOP 20 MOST IMPORTANT FEATURES (Random Forest)")
    print("="*60)
    importance_df = get_feature_importance(rf_model, feature_names, top_n=20)
    if importance_df is not None:
        print(importance_df.to_string(index=False))
    
    # Save model and preprocessing objects
    print("\n" + "="*60)
    print("Saving model and preprocessing objects...")
    print("="*60)
    joblib.dump(rf_model, 'classification_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Model saved to: classification_model.pkl")
    print("Preprocessing objects saved.")
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('classification_results.csv')
    print("Results saved to: classification_results.csv")
    
    if importance_df is not None:
        importance_df.to_csv('feature_importance.csv', index=False)
        print("Feature importance saved to: feature_importance.csv")
    
    print("\n" + "="*60)
    print("CLASSIFICATION MODEL TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

