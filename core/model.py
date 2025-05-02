import pandas as pd
import numpy as np
import joblib

from core.loader import load_cpt_files, split_cpt_files
from core.classifier import train_soil_model, predict_soil_types
from core.interpolator import create_3d_interpolation, create_3d_interpolation_alternative
from utils.soil_types import SoilTypeManager
from utils.helpers import get_compatible_colormap
from visualization.plots import (
    plot_cpt_locations, 
    plot_cpt_profile, 
    plot_feature_importance
)
from visualization.model_3d import visualize_3d_model
from visualization.evaluation import plot_test_vs_predicted


class CPT_3D_SoilModel:
    """
    Framework for creating 3D soil models from CPT data using machine learning.
    """
    
    def __init__(self):
        """Initialize the framework components"""
        self.cpt_data = None
        self.train_data = None
        self.test_data = None
        self.soil_model = None
        self.scaler = None
        self.feature_columns = None
        self.soil_types = None
        self.soil_colors = None
        self.interpolator = None
        self.soil_manager = SoilTypeManager()
    
    def load_data(self, file_pattern, x_coord_col=None, y_coord_col=None, test_size=0.2, random_state=42):
        """
        Load CPT data from multiple files and combine with spatial coordinates.
        Split the data between training and testing based on CPT files.
        
        Parameters:
        -----------
        file_pattern : str
            File pattern to match CPT files (e.g., "data/CPT_*.csv")
        x_coord_col : str, optional
            Column name containing X coordinates (Easting)
        y_coord_col : str, optional
            Column name containing Y coordinates (Northing)
        test_size : float
            Proportion of CPT files to use for testing (default: 0.2)
        random_state : int
            Random seed for reproducibility
        """
        print(f"Loading data from pattern: {file_pattern}")
        
        # Split files into train and test
        train_files, test_files = split_cpt_files(file_pattern, test_size, random_state)
        
        print(f"Split data: {len(train_files)} files for training, {len(test_files)} files for testing")
        
        # Load and combine data
        train_data = load_cpt_files(train_files, x_coord_col, y_coord_col, is_train=True)
        test_data = load_cpt_files(test_files, x_coord_col, y_coord_col, is_train=False)
        
        # Store the data
        self.train_data = train_data
        self.test_data = test_data
        self.cpt_data = pd.concat([train_data, test_data], ignore_index=True)
        
        print(f"Loaded {len(train_data)} training records and {len(test_data)} testing records")
        
        # Set default feature columns if soil column exists
        if 'soil []' in self.cpt_data.columns:
            self.feature_columns = ['qc [MPa]', 'fs [MPa]', 'Rf [%]', 'u2 [MPa]']
            self.soil_types = sorted(self.cpt_data['soil []'].unique())
            
            # Create a color map for soil types
            cmap = get_compatible_colormap('viridis', len(self.soil_types))
            
            # Create a soil_labels dictionary
            soil_labels = SoilTypeManager.create_label_map()
            
            # Create color dictionary with labels that include abbreviations
            self.soil_colors = {}
            for soil_type, (r,g,b,_) in zip(self.soil_types, 
                                           [cmap(i) for i in range(len(self.soil_types))]):
                key = soil_type
                label = soil_labels.get(soil_type, f"{soil_type}")
                self.soil_colors[key] = {
                    'color': f'rgb({int(255*r)},{int(255*g)},{int(255*b)})',
                    'label': label
                }
        
        return self.train_data, self.test_data

    def explore_data(self, show_dataset_overview=True, show_soil_distribution=True, show_cpt_locations=True, show_cpt_profiles=True):
        """
        Explore and visualize the loaded data
        
        Parameters:
        -----------
        show_dataset_overview : bool
            Whether to display dataset overview
        show_soil_distribution : bool
            Whether to show soil type distribution
        show_cpt_locations : bool
            Whether to plot CPT locations
        show_cpt_profiles : bool
            Whether to plot CPT profiles
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Basic information
        if show_dataset_overview:
            print("\nDataset Overview:")
            print(f"Total records: {len(self.cpt_data)}")
            print(f"Training records: {len(self.train_data)} ({len(self.train_data)/len(self.cpt_data)*100:.1f}%)")
            print(f"Testing records: {len(self.test_data)} ({len(self.test_data)/len(self.cpt_data)*100:.1f}%)")
            print(f"CPT locations: {self.cpt_data['cpt_id'].nunique()}")
            print(f"Training CPT locations: {self.train_data['cpt_id'].nunique()}")
            print(f"Testing CPT locations: {self.test_data['cpt_id'].nunique()}")
            print(f"Columns: {', '.join(self.cpt_data.columns)}")
        
        # Check for soil classification column
        if 'soil []' in self.cpt_data.columns and show_soil_distribution:
            soil_counts = self.cpt_data['soil []'].value_counts()
            train_soil_counts = self.train_data['soil []'].value_counts()
            test_soil_counts = self.test_data['soil []'].value_counts()
            
            print("\nSoil Type Distribution (Total Dataset):")
            for soil_type, count in soil_counts.items():
                abbr = SoilTypeManager.get_abbreviation(soil_type)
                print(f"  Type {soil_type} ({abbr}): {count} records ({count/len(self.cpt_data)*100:.1f}%)")
            
            print("\nSoil Type Distribution (Training Set):")
            for soil_type in sorted(train_soil_counts.index):
                count = train_soil_counts.get(soil_type, 0)
                abbr = SoilTypeManager.get_abbreviation(soil_type)
                print(f"  Type {soil_type} ({abbr}): {count} records ({count/len(self.train_data)*100:.1f}%)")
            
            print("\nSoil Type Distribution (Testing Set):")
            for soil_type in sorted(test_soil_counts.index):
                count = test_soil_counts.get(soil_type, 0)
                abbr = SoilTypeManager.get_abbreviation(soil_type)
                print(f"  Type {soil_type} ({abbr}): {count} records ({count/len(self.test_data)*100:.1f}%)")
        
        # Plot CPT locations
        if show_cpt_locations:
            plot_cpt_locations(self.cpt_data, show_train_test=True)
        
        # Plot example CPT profile
        if show_cpt_profiles:
            # Plot one example from training set
            if len(self.train_data['cpt_id'].unique()) > 0:
                train_example = self.train_data['cpt_id'].unique()[0]
                print(f"\nPlotting example CPT profile from training set: {train_example}")
                plot_cpt_profile(self.train_data, train_example)
            
            # Plot one example from testing set
            if len(self.test_data['cpt_id'].unique()) > 0:
                test_example = self.test_data['cpt_id'].unique()[0]
                print(f"\nPlotting example CPT profile from testing set: {test_example}")
                plot_cpt_profile(self.test_data, test_example)
    
    def train_soil_classification_model(self, test_size=0.2, random_state=42, model_type='rf'):
        """
        Train a machine learning model to predict soil type from CPT parameters
        using the training data set only
        
        Parameters:
        -----------
        test_size : float
            Proportion of training data to use for validation
        random_state : int
            Random seed for reproducibility
        model_type : str
            Type of model to train ('rf' for Random Forest, 'xgb' for XGBoost)
        """
        if self.train_data is None:
            raise ValueError("No training data loaded. Call load_data() first.")
        
        if 'soil []' not in self.train_data.columns:
            raise ValueError("No soil classification column found in the training data")
        
        print("Training soil classification model using only training data...")
        
        # Define features and target
        if self.feature_columns is None:
            # Default feature columns if not specified
            self.feature_columns = ['qc [MPa]', 'fs [MPa]', 'Rf [%]', 'u2 [MPa]']
        
        # Train the model
        self.soil_model, self.scaler, validation_results = train_soil_model(
            self.train_data, 
            self.feature_columns,
            test_size=test_size,
            random_state=random_state,
            model_type=model_type
        )
        
        # Display validation results
        val_accuracy = validation_results['accuracy']
        val_report = validation_results['report']
        val_y_pred = validation_results['y_pred']
        val_y_true = validation_results['y_true']
        best_params = validation_results['best_params']
        
        print("\nModel Evaluation on Validation Set (Split from Train):")
        print(f"Best parameters: {best_params}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print("\nClassification Report on Validation Set:")
        print(val_report)
        
        # Stampa intestazioni con abbreviazioni
        print("\nValidation Results by Soil Type:")
        for soil_type in sorted(set(val_y_true)):
            abbr = SoilTypeManager.get_abbreviation(soil_type)
            desc = SoilTypeManager.get_description(soil_type)
            mask = (val_y_true == soil_type)
            accuracy = (val_y_pred[mask] == soil_type).mean() if mask.sum() > 0 else 0
            print(f"  Type {soil_type} ({abbr}) - {desc}: Accuracy {accuracy:.4f}")
        
        # Feature importance
        if hasattr(self.soil_model, 'feature_importances_'):
            plot_feature_importance(self.soil_model, self.feature_columns)
        
        return self.soil_model
    
    def evaluate_on_test_data(self):
        """
        Evaluate the trained model on the separate test data set
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.test_data is None:
            raise ValueError("No test data loaded. Call load_data() first.")
            
        if self.soil_model is None:
            raise ValueError("No model trained. Call train_soil_classification_model() first.")
        
        print("\nEvaluating model on separate test CPT files...")
        
        # Extract features and target from test data
        X_test = self.test_data[self.feature_columns].copy()
        y_test = self.test_data['soil []']
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict soil types
        y_pred_test = self.soil_model.predict(X_test_scaled)
        
        # Calculate overall accuracy
        test_accuracy = (y_pred_test == y_test).mean()
        
        print(f"Test set accuracy: {test_accuracy:.4f}")
        print("\nClassification Report on Test Set:")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred_test))
        
        # Evaluate by soil type
        print("\nTest Results by Soil Type:")
        soil_type_metrics = {}
        for soil_type in sorted(set(y_test)):
            abbr = SoilTypeManager.get_abbreviation(soil_type)
            desc = SoilTypeManager.get_description(soil_type)
            mask = (y_test == soil_type)
            count = mask.sum()
            if count > 0:
                accuracy = (y_pred_test[mask] == soil_type).mean()
                soil_type_metrics[soil_type] = {
                    'count': count,
                    'accuracy': accuracy,
                    'abbr': abbr,
                    'desc': desc
                }
                print(f"  Type {soil_type} ({abbr}) - {desc}: Accuracy {accuracy:.4f} ({count} samples)")
            else:
                print(f"  Type {soil_type} ({abbr}) - {desc}: No samples in test set")
        
        # Evaluate by CPT file
        print("\nTest Results by CPT File:")
        cpt_metrics = {}
        for cpt_id in self.test_data['cpt_id'].unique():
            mask = (self.test_data['cpt_id'] == cpt_id)
            cpt_y_test = y_test[mask]
            cpt_y_pred = y_pred_test[mask]
            accuracy = (cpt_y_pred == cpt_y_test).mean()
            cpt_metrics[cpt_id] = {
                'count': mask.sum(),
                'accuracy': accuracy
            }
            print(f"  CPT {cpt_id}: Accuracy {accuracy:.4f} ({mask.sum()} samples)")
        
        # Visualize test vs predicted
        plot_test_vs_predicted(y_test, y_pred_test, soil_types=self.soil_types)
        
        # Add predictions to the test data
        self.test_data['predicted_soil'] = y_pred_test
        self.test_data['predicted_soil_abbr'] = self.test_data['predicted_soil'].apply(SoilTypeManager.get_abbreviation)
        self.test_data['predicted_soil_desc'] = self.test_data['predicted_soil'].apply(SoilTypeManager.get_description)
        
        # Return metrics
        return {
            'overall_accuracy': test_accuracy,
            'soil_type_metrics': soil_type_metrics,
            'cpt_metrics': cpt_metrics
        }
    
    def predict_soil_types(self):
        """
        Predict soil types for CPT data points using the trained model
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
                
        if self.soil_model is None:
            raise ValueError("No model trained. Call train_soil_classification_model() first.")
        
        print("Predicting soil types for all data...")
        
        # Predict on all data
        from core.classifier import predict_soil_types as predict_func
        self.cpt_data = predict_func(
            self.cpt_data,
            self.soil_model,
            self.scaler,
            self.feature_columns
        )
        
        # Importante: aggiungi le colonne predicted anche ai dataframe train e test
        # usando l'identificatore cpt_id per il matching
        print("Updating train and test dataframes with predictions...")
        
        # Per ogni CPT nel set di train
        for cpt_id in self.train_data['cpt_id'].unique():
            # Ottieni le predizioni dal dataframe completo
            mask_cpt = self.cpt_data['cpt_id'] == cpt_id
            pred_soil = self.cpt_data.loc[mask_cpt, 'predicted_soil'].values
            pred_abbr = self.cpt_data.loc[mask_cpt, 'predicted_soil_abbr'].values
            pred_desc = self.cpt_data.loc[mask_cpt, 'predicted_soil_desc'].values
            
            # Aggiungi le predizioni al dataframe di train
            mask_train = self.train_data['cpt_id'] == cpt_id
            self.train_data.loc[mask_train, 'predicted_soil'] = pred_soil
            self.train_data.loc[mask_train, 'predicted_soil_abbr'] = pred_abbr
            self.train_data.loc[mask_train, 'predicted_soil_desc'] = pred_desc
        
        # Per ogni CPT nel set di test
        for cpt_id in self.test_data['cpt_id'].unique():
            # Ottieni le predizioni dal dataframe completo
            mask_cpt = self.cpt_data['cpt_id'] == cpt_id
            pred_soil = self.cpt_data.loc[mask_cpt, 'predicted_soil'].values
            pred_abbr = self.cpt_data.loc[mask_cpt, 'predicted_soil_abbr'].values
            pred_desc = self.cpt_data.loc[mask_cpt, 'predicted_soil_desc'].values
            
            # Aggiungi le predizioni al dataframe di test
            mask_test = self.test_data['cpt_id'] == cpt_id
            self.test_data.loc[mask_test, 'predicted_soil'] = pred_soil
            self.test_data.loc[mask_test, 'predicted_soil_abbr'] = pred_abbr
            self.test_data.loc[mask_test, 'predicted_soil_desc'] = pred_desc
        
        # Verifica che le colonne siano state aggiunte correttamente
        print(f"Train data shape: {self.train_data.shape}, 'predicted_soil' in columns: {'predicted_soil' in self.train_data.columns}")
        print(f"Test data shape: {self.test_data.shape}, 'predicted_soil' in columns: {'predicted_soil' in self.test_data.columns}")
        
        # Calcola accuratezza 
        if 'soil []' in self.cpt_data.columns:
            train_mask = self.cpt_data['is_train'] == True
            test_mask = self.cpt_data['is_train'] == False
            
            train_accuracy = (self.cpt_data.loc[train_mask, 'predicted_soil'] == 
                            self.cpt_data.loc[train_mask, 'soil []']).mean()
            print(f"Training data prediction accuracy: {train_accuracy:.4f}")
            
            test_accuracy = (self.cpt_data.loc[test_mask, 'predicted_soil'] == 
                            self.cpt_data.loc[test_mask, 'soil []']).mean()
            print(f"Test data prediction accuracy: {test_accuracy:.4f}")
        
        return self.cpt_data['predicted_soil']
    
    def create_3d_interpolation(self, resolution=10, use_test_data=False):
        """
        Create a 3D interpolation of soil types from discrete CPT points
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Verifica se la colonna 'predicted_soil' è presente nei dataframe
        if 'predicted_soil' not in self.cpt_data.columns:
            raise ValueError("No soil predictions in cpt_data. Call predict_soil_types() first.")
        
        # Verifica nel train_data
        if 'predicted_soil' not in self.train_data.columns:
            raise ValueError("No soil predictions in train_data. There's an issue with predict_soil_types().")
        
        # Seleziona i dati da utilizzare per l'interpolazione
        if use_test_data:
            # Usa tutti i dati
            data_to_use = self.cpt_data  
            print(f"Using all data for interpolation ({len(data_to_use)} records)")
        else:
            # Usa solo i dati di training
            data_to_use = self.train_data
            print(f"Using only training data for interpolation ({len(data_to_use)} records)")
        
        # Verifica la presenza delle colonne necessarie
        required_cols = ['x_coord', 'y_coord', data_to_use.columns[0], 'predicted_soil']
        for col in required_cols:
            if col not in data_to_use.columns:
                raise ValueError(f"Required column '{col}' not found in data_to_use")
        
        # Stampa le prime righe per debug
        print("First 5 rows of data_to_use:")
        print(data_to_use[required_cols].head())
        
        # Crea l'interpolazione
        from core.interpolator import create_3d_interpolation as interp_func
        interp_result = interp_func(
            data_to_use,
            resolution=resolution,
            is_train_only=not use_test_data
        )
        
        # Salva l'interpolatore
        self.interpolator = interp_result['interpolator']
        
        return interp_result['grid_data']
    
    def create_3d_interpolation_alternative(self, resolution=10, use_test_data=False):
        """
        Create a 3D interpolation of soil types using nearest neighbor method
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Verifica se la colonna 'predicted_soil' è presente nei dataframe
        if 'predicted_soil' not in self.cpt_data.columns:
            raise ValueError("No soil predictions in cpt_data. Call predict_soil_types() first.")
        
        # Verifica nel train_data
        if 'predicted_soil' not in self.train_data.columns:
            raise ValueError("No soil predictions in train_data. There's an issue with predict_soil_types().")
        
        # Seleziona i dati da utilizzare per l'interpolazione
        if use_test_data:
            # Usa tutti i dati
            data_to_use = self.cpt_data  
            print(f"Using all data for interpolation ({len(data_to_use)} records)")
        else:
            # Usa solo i dati di training
            data_to_use = self.train_data
            print(f"Using only training data for interpolation ({len(data_to_use)} records)")
        
        # Verifica la presenza delle colonne necessarie
        required_cols = ['x_coord', 'y_coord', data_to_use.columns[0], 'predicted_soil']
        for col in required_cols:
            if col not in data_to_use.columns:
                raise ValueError(f"Required column '{col}' not found in data_to_use")
        
        # Stampa le prime righe per debug
        print("First 5 rows of data_to_use:")
        print(data_to_use[required_cols].head())
        
        # Crea l'interpolazione
        from core.interpolator import create_3d_interpolation_alternative as interp_func
        interp_result = interp_func(
            data_to_use,
            resolution=resolution,
            is_train_only=not use_test_data
        )
        
        # Salva l'interpolatore
        self.interpolator = interp_result['interpolator']
        
        return interp_result['grid_data']
    
    def visualize_3d_model(self, interpolation_data=None, interactive=True, use_test_data=False):
        """
        Visualize the 3D soil model
        
        Parameters:
        -----------
        interpolation_data : dict, optional
            Output from create_3d_interpolation()
        interactive : bool
            Whether to create an interactive Plotly visualization
        use_test_data : bool
            Whether to include test data in the visualization
        """
        if interpolation_data is None:
            interpolation_data = self.create_3d_interpolation(use_test_data=use_test_data)
            
        # Choose which dataset to visualize
        data_to_visualize = self.cpt_data if use_test_data else self.train_data
        
        return visualize_3d_model(
            data_to_visualize,
            interpolation_data,
            self.soil_types,
            self.soil_colors,
            interactive
        )
    
    def save_model(self, filename='cpt_soil_model.pkl'):
        """
        Save the trained model and related components
        
        Parameters:
        -----------
        filename : str
            Filename to save the model
        """
        if self.soil_model is None:
            raise ValueError("No model to save. Train a model first.")
        
        model_data = {
            'soil_model': self.soil_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'soil_types': self.soil_types,
            'soil_colors': self.soil_colors
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='cpt_soil_model.pkl'):
        """
        Load a previously saved model
        
        Parameters:
        -----------
        filename : str
            Filename to load the model from
        """
        model_data = joblib.load(filename)
        
        self.soil_model = model_data['soil_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.soil_types = model_data['soil_types']
        self.soil_colors = model_data['soil_colors']
        
        print(f"Model loaded from {filename}")