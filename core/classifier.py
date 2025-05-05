import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    
    # Analyze class distribution
    _analyze_class_distribution(y)
    
    # Calculate class weights inversely proportional to class frequencies
    class_weights = _calculate_class_weights(y)
    
    # Create label encoder for XGBoost if needed
    label_encoder = None
    original_classes = None
    if model_type.lower() == 'xgb':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        original_classes = label_encoder.classes_
        print("\nLabel encoding for XGBoost:")
        for i, label in enumerate(original_classes):
            print(f"  Original soil type {label} -> Encoded as {i}")
    
    # Split data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply resampling techniques per applicare SMOTE o sampling
    try:
        from imblearn.over_sampling import SMOTE, RandomOverSampler
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        print("\nApplying resampling techniques to balance classes...")
        
        # Creiamo una pipeline di resampling
        # Prima SMOTE per le classi minoritarie, poi undersampling per ridurre le classi maggioritarie
        # Questo approccio è spesso più efficace del solo SMOTE
        over = SMOTE(sampling_strategy='auto', random_state=random_state)
        under = RandomUnderSampler(sampling_strategy='auto', random_state=random_state)
        
        steps = [('over', over), ('under', under)]
        pipeline = Pipeline(steps=steps)
        
        # Applichiamo la pipeline
        X_train_scaled, y_train = pipeline.fit_resample(X_train_scaled, y_train)
        
        print(f"After resampling: {len(X_train_scaled)} samples")
        y_train_counts = pd.Series(y_train).value_counts()
        print("New class distribution after resampling:")
        for cls, count in sorted(y_train_counts.items()):
            print(f"  Class {cls}: {count} samples")
            
    except ImportError:
        print("imblearn not installed. Skipping resampling. Install with: pip install imbalanced-learn")
    
    # Create and configure model based on type
    model, param_grid, sample_weights = _create_model_config(
        model_type, y_train, class_weights, random_state
    )
    
    # Train model with grid search
    model = _train_model_with_grid_search(
        model, param_grid, X_train_scaled, y_train, 
        sample_weights, model_type
    )
    
    # For XGBoost, we need to store the label encoder for later use
    if model_type.lower() == 'xgb':
        model.label_encoder_ = label_encoder
        model.original_classes_ = original_classes
    
    # Evaluate model - need to handle the case where we used a label encoder
    results = _evaluate_model(model, X_val_scaled, y_val, model_type, label_encoder)
    
    return model, scaler, results


def _analyze_class_distribution(y):
    """
    Analyze and print the class distribution
    
    Parameters:
    -----------
    y : Series
        Target variable
    """
    class_distribution = y.value_counts()
    print("\nClass distribution in training data:")
    for soil_type, count in class_distribution.items():
        abbr = SoilTypeManager.get_abbreviation(soil_type)
        print(f"  Type {soil_type} ({abbr}): {count} records ({count/len(y)*100:.1f}%)")


def _calculate_class_weights(y):
    """
    Calculate class weights inversely proportional to class frequencies
    
    Parameters:
    -----------
    y : Series
        Target variable
        
    Returns:
    --------
    dict
        Dictionary of class weights
    """
    class_distribution = y.value_counts()
    class_weights = {cls: len(y) / (len(class_distribution) * count) 
                    for cls, count in class_distribution.items()}
    
    print("\nCalculated class weights:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls}: {weight:.2f}")
    
    return class_weights


def _create_model_config(model_type, y_train, class_weights, random_state):
    """
    Create model, parameter grid, and sample weights based on model type
    """
    sample_weights = None
    
    if model_type.lower() == 'rf':
        # Random Forest model with class balancing
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 15, 30],
            'min_samples_split': [2, 5],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        model = RandomForestClassifier(random_state=random_state)
        
    elif model_type.lower() == 'xgb':
        # XGBoost model - removing scale_pos_weight parameter
        # Prepare sample weights for class balancing
        sample_weights = np.ones(len(y_train))
        unique_classes = np.unique(y_train)
        for i, cls in enumerate(unique_classes):
            weight_idx = np.where(y_train == cls)[0]
            weight_value = 1 / (len(unique_classes) * (len(weight_idx) / len(y_train)))
            sample_weights[weight_idx] = weight_value
            
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 8, 12],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 1.0],  # Controls sample percentage per tree
            'colsample_bytree': [0.8, 1.0]  # Controls feature percentage per tree
            # scale_pos_weight parameter removed
        }
        
        model = xgb.XGBClassifier(random_state=random_state)
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, param_grid, sample_weights


def _train_model_with_grid_search(model, param_grid, X_train, y_train, sample_weights, model_type):
    """
    Train model using grid search
    
    Parameters:
    -----------
    model : sklearn model
        Base model
    param_grid : dict
        Parameter grid for grid search
    X_train : array
        Training features
    y_train : array
        Training targets
    sample_weights : array
        Sample weights (can be None)
    model_type : str
        Model type
        
    Returns:
    --------
    model : sklearn model
        Trained model
    """
    print("\nStarting grid search with class balancing...")
    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring='balanced_accuracy', n_jobs=-1
    )
    
    if model_type.lower() == 'xgb' and sample_weights is not None:
        # Per XGBoost, usa i pesi campione
        grid_search.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_


def _evaluate_model(model, X_val, y_val, model_type='rf', label_encoder=None):
    """
    Evaluate model on validation set
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_val : array
        Validation features
    y_val : array
        Validation targets
    model_type : str
        Type of model ('rf' or 'xgb')
    label_encoder : LabelEncoder, optional
        Label encoder used for XGBoost
        
    Returns:
    --------
    dict
        Dictionary with evaluation results
    """
    # Predict on validation set
    y_pred_val = model.predict(X_val)
    
    # For XGBoost, transform predictions back to original labels for evaluation
    if model_type.lower() == 'xgb' and label_encoder is not None:
        y_pred_val_original = label_encoder.inverse_transform(y_pred_val.astype(int))
        y_val_original = label_encoder.inverse_transform(y_val.astype(int))
        
        # Calculate accuracy on original labels
        val_accuracy = (y_pred_val_original == y_val_original).mean()
        
        # Generate classification report on original labels with zero_division=0
        # per evitare i warning quando non ci sono campioni predetti per una classe
        val_report = classification_report(y_val_original, y_pred_val_original, 
                                          zero_division=0)
        
        # Collect results
        results = {
            'accuracy': val_accuracy,
            'report': val_report,
            'y_pred': y_pred_val_original,
            'y_true': y_val_original,
            'best_params': model.get_params()
        }
    else:
        # Calculate accuracy
        val_accuracy = model.score(X_val, y_val)
        
        # Generate classification report with zero_division=0
        val_report = classification_report(y_val, y_pred_val, zero_division=0)
        
        # Collect results
        results = {
            'accuracy': val_accuracy,
            'report': val_report,
            'y_pred': y_pred_val,
            'y_true': y_val,
            'best_params': model.get_params()
        }
    
    return results


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
    predictions = model.predict(X_scaled)
    
    # If the model is XGBoost, we need to transform predictions back to original labels
    if hasattr(model, 'label_encoder_'):
        predictions = model.label_encoder_.inverse_transform(predictions.astype(int))
    
    data['predicted_soil'] = predictions
    
    # Add abbreviations and descriptions for the predicted soil types
    data['predicted_soil_abbr'] = data['predicted_soil'].apply(SoilTypeManager.get_abbreviation)
    data['predicted_soil_desc'] = data['predicted_soil'].apply(SoilTypeManager.get_description)
    
    return data