import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import xgboost as xgb

from utils.soil_types import SoilTypeManager


def train_soil_model(train_data, feature_columns, test_size=0.2, random_state=42, model_type='rf'):
    """
    Train a machine learning model to predict soil type from CPT parameters
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data containing CPT parameters and soil types
    feature_columns : list
        List of column names to use as features
    test_size : float
        Proportion of data to use for validation
    random_state : int
        Random seed for reproducibility
    model_type : str
        Type of model to train ('rf' for Random Forest, 'xgb' for XGBoost)
        
    Returns:
    --------
    model : sklearn model
        Trained soil classification model
    scaler : StandardScaler
        Fitted feature scaler
    results : dict
        Dictionary containing validation results
    """
    # Extract features and target
    X = train_data[feature_columns].copy()
    y = train_data['soil []']
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    if model_type.lower() == 'rf':
        # Random Forest model
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        base_model = RandomForestClassifier(random_state=random_state)
        
    elif model_type.lower() == 'xgb':
        # XGBoost model
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.1, 0.2]
        }
        
        base_model = xgb.XGBClassifier(random_state=random_state)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Use grid search to find best parameters
    grid_search = GridSearchCV(
        base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    # Get best model
    model = grid_search.best_estimator_
    
    # Evaluate model on validation set
    y_pred_val = model.predict(X_val_scaled)
    val_accuracy = model.score(X_val_scaled, y_val)
    val_report = classification_report(y_val, y_pred_val)
    
    # Collect results
    results = {
        'accuracy': val_accuracy,
        'report': val_report,
        'y_pred': y_pred_val,
        'y_true': y_val,
        'best_params': grid_search.best_params_
    }
    
    return model, scaler, results


def predict_soil_types(data, model, scaler, feature_columns):
    """
    Predict soil types for CPT data points using the trained model
    
    Parameters:
    -----------
    data : pandas.DataFrame
        CPT data to predict soil types for
    model : sklearn model
        Trained soil classification model
    scaler : StandardScaler
        Fitted feature scaler
    feature_columns : list
        List of column names to use as features
        
    Returns:
    --------
    data : pandas.DataFrame
        Input data with added soil type predictions
    """
    # Extract features
    X = data[feature_columns].copy()
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict soil types
    data['predicted_soil'] = model.predict(X_scaled)
    
    # Add abbreviations and descriptions for the predicted soil types
    data['predicted_soil_abbr'] = data['predicted_soil'].apply(SoilTypeManager.get_abbreviation)
    data['predicted_soil_desc'] = data['predicted_soil'].apply(SoilTypeManager.get_description)
    
    return data