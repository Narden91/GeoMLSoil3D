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
from visualization.model_3d import visualize_3d_model, visualize_compare_3d_models
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
    
    def load_data(self, file_pattern, x_coord_col=None, y_coord_col=None, test_size=0.2, random_state=42,
             auto_detect=True, encoding='utf-8', separator=',', header=0, depth_col=None):
        """
        Carica dati CPT da più file e li combina con coordinate spaziali.
        Divide i dati tra training e testing basandosi sui file CPT.
        
        Parameters:
        -----------
        file_pattern : str
            Pattern per trovare i file CPT (es. "data/CPT_*.csv")
        x_coord_col : str, optional
            Nome della colonna contenente le coordinate X
        y_coord_col : str, optional
            Nome della colonna contenente le coordinate Y
        test_size : float
            Proporzione di file CPT da usare per il testing (default: 0.2)
        random_state : int
            Seed per riproducibilità
        auto_detect : bool
            Rileva automaticamente il formato CSV
        encoding : str
            Codifica del file (se auto_detect=False)
        separator : str
            Separatore CSV (se auto_detect=False)
        header : int
            Riga da usare come intestazione (se auto_detect=False)
        depth_col : str, optional
            Nome della colonna contenente i valori di profondità
        """
        print(f"Caricamento dati dal pattern: {file_pattern}")
        
        # Dividi i file in train e test
        train_files, test_files = split_cpt_files(file_pattern, test_size, random_state)
        
        print(f"Dati divisi: {len(train_files)} file per training, {len(test_files)} file per testing")
        
        # Carica e combina i dati
        train_data = load_cpt_files(
            train_files, x_coord_col, y_coord_col, is_train=True,
            auto_detect=auto_detect, encoding=encoding, separator=separator,
            header=header, depth_col=depth_col
        )
        
        test_data = load_cpt_files(
            test_files, x_coord_col, y_coord_col, is_train=False,
            auto_detect=auto_detect, encoding=encoding, separator=separator,
            header=header, depth_col=depth_col
        )
        
        # Memorizza i dati
        self.train_data = train_data
        self.test_data = test_data
        self.cpt_data = pd.concat([train_data, test_data], ignore_index=True)
        
        print(f"Caricati {len(train_data)} record di training e {len(test_data)} record di testing")
        
        # Inizializza tipi di suolo e colori se possibile
        self._initialize_soil_types_and_colors()
        
        return self.train_data, self.test_data
    
    def _initialize_soil_types_and_colors(self):
        """Initialize soil types and colors based on available data"""
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
            self._display_dataset_overview()
        
        # Check for soil classification column
        if 'soil []' in self.cpt_data.columns and show_soil_distribution:
            self._display_soil_distribution()
        
        # Plot CPT locations
        if show_cpt_locations:
            plot_cpt_locations(self.cpt_data, show_train_test=True)
        
        # Plot example CPT profiles
        if show_cpt_profiles:
            self._plot_example_profiles()
    
    def _display_dataset_overview(self):
        """Display basic dataset overview information"""
        print("\nDataset Overview:")
        print(f"Total records: {len(self.cpt_data)}")
        print(f"Training records: {len(self.train_data)} ({len(self.train_data)/len(self.cpt_data)*100:.1f}%)")
        print(f"Testing records: {len(self.test_data)} ({len(self.test_data)/len(self.cpt_data)*100:.1f}%)")
        print(f"CPT locations: {self.cpt_data['cpt_id'].nunique()}")
        print(f"Training CPT locations: {self.train_data['cpt_id'].nunique()}")
        print(f"Testing CPT locations: {self.test_data['cpt_id'].nunique()}")
        print(f"Columns: {', '.join(self.cpt_data.columns)}")
    
    def _display_soil_distribution(self):
        """Display soil type distribution in datasets"""
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
    
    def _plot_example_profiles(self):
        """Plot example CPT profiles from training and testing sets"""
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
        self._display_validation_results(validation_results)
        
        # Plot feature importance if available
        if hasattr(self.soil_model, 'feature_importances_'):
            plot_feature_importance(self.soil_model, self.feature_columns)
        
        return self.soil_model
    
    def _display_validation_results(self, validation_results):
        """
        Display validation results
        
        Parameters:
        -----------
        validation_results : dict
            Dictionary with validation results
        """
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
        
        # Extract features and target
        X_test = self.test_data[self.feature_columns].copy()
        y_test = self.test_data['soil []']
        
        # Scale features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict soil types
        y_pred_test = self.soil_model.predict(X_test_scaled)
        
        # Calculate metrics
        test_metrics = self._calculate_test_metrics(y_test, y_pred_test)
        
        # Visualize test vs predicted
        plot_test_vs_predicted(y_test, y_pred_test, soil_types=self.soil_types)
        
        # Add predictions to the test data
        self.test_data['predicted_soil'] = y_pred_test
        self.test_data['predicted_soil_abbr'] = self.test_data['predicted_soil'].apply(SoilTypeManager.get_abbreviation)
        self.test_data['predicted_soil_desc'] = self.test_data['predicted_soil'].apply(SoilTypeManager.get_description)
        
        return test_metrics
    
    def _calculate_test_metrics(self, y_test, y_pred_test):
        """
        Calculate metrics for test data
        
        Parameters:
        -----------
        y_test : array
            True soil types
        y_pred_test : array
            Predicted soil types
            
        Returns:
        --------
        dict
            Dictionary with test metrics
        """
        from sklearn.metrics import classification_report
        
        # Calculate overall accuracy
        test_accuracy = (y_pred_test == y_test).mean()
        
        print(f"Test set accuracy: {test_accuracy:.4f}")
        print("\nClassification Report on Test Set:")
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
        
        # Update train and test dataframes with predictions
        self._update_train_test_with_predictions()
        
        # Calculate accuracy if soil types are available
        self._calculate_prediction_accuracy()
        
        return self.cpt_data['predicted_soil']
    
    def _update_train_test_with_predictions(self):
        """Update training and testing dataframes with predictions"""
        print("Updating train and test dataframes with predictions...")
        
        # Update training dataframe
        self._update_dataframe_with_predictions(self.train_data)
        
        # Update testing dataframe
        self._update_dataframe_with_predictions(self.test_data)
        
        # Verify updates
        print(f"Train data shape: {self.train_data.shape}, 'predicted_soil' in columns: {'predicted_soil' in self.train_data.columns}")
        print(f"Test data shape: {self.test_data.shape}, 'predicted_soil' in columns: {'predicted_soil' in self.test_data.columns}")
    
    def _update_dataframe_with_predictions(self, df):
        """
        Update a dataframe with predictions from the main dataframe
        
        Parameters:
        -----------
        df : DataFrame
            Dataframe to update
        """
        for cpt_id in df['cpt_id'].unique():
            # Get predictions from main dataframe
            mask_cpt = self.cpt_data['cpt_id'] == cpt_id
            pred_soil = self.cpt_data.loc[mask_cpt, 'predicted_soil'].values
            pred_abbr = self.cpt_data.loc[mask_cpt, 'predicted_soil_abbr'].values
            pred_desc = self.cpt_data.loc[mask_cpt, 'predicted_soil_desc'].values
            
            # Add predictions to the dataframe
            mask_df = df['cpt_id'] == cpt_id
            df.loc[mask_df, 'predicted_soil'] = pred_soil
            df.loc[mask_df, 'predicted_soil_abbr'] = pred_abbr
            df.loc[mask_df, 'predicted_soil_desc'] = pred_desc
    
    def _calculate_prediction_accuracy(self):
        """Calculate and display prediction accuracy if real soil types are available"""
        if 'soil []' in self.cpt_data.columns:
            train_mask = self.cpt_data['is_train'] == True
            test_mask = self.cpt_data['is_train'] == False
            
            train_accuracy = (self.cpt_data.loc[train_mask, 'predicted_soil'] == 
                            self.cpt_data.loc[train_mask, 'soil []']).mean()
            print(f"Training data prediction accuracy: {train_accuracy:.4f}")
            
            test_accuracy = (self.cpt_data.loc[test_mask, 'predicted_soil'] == 
                            self.cpt_data.loc[test_mask, 'soil []']).mean()
            print(f"Test data prediction accuracy: {test_accuracy:.4f}")
            
    def create_3d_interpolation(self, resolution=10, use_test_data=False, soil_col='predicted_soil', method='linear'):
        """
        Create a 3D interpolation of soil types from discrete CPT points
        
        Parameters:
        -----------
        resolution : int
            Resolution of the interpolation grid
        use_test_data : bool
            Whether to include test data in the interpolation
        soil_col : str
            Column name containing soil types to interpolate
        method : str
            Interpolation method ('linear' or 'nearest')
            
        Returns:
        --------
        dict
            Grid data for 3D visualization
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Verifica se la colonna del suolo specificata è presente nei dataframe
        if soil_col not in self.cpt_data.columns:
            raise ValueError(f"No soil column '{soil_col}' in cpt_data. Call predict_soil_types() first or use 'soil []' for real values.")
        
        # Seleziona i dati da utilizzare per l'interpolazione
        if use_test_data:
            # Usa tutti i dati
            data_to_use = self.cpt_data.copy()  
            print(f"Using all data for interpolation with {soil_col} column ({len(data_to_use)} records)")
        else:
            # Usa solo i dati di training
            data_to_use = self.train_data.copy()
            print(f"Using only training data for interpolation with {soil_col} column ({len(data_to_use)} records)")
        
        # Verifica che la colonna del suolo sia presente nei dati da usare
        if soil_col not in data_to_use.columns:
            raise ValueError(f"Required column '{soil_col}' not found in data_to_use")
        
        # Verifica la presenza delle colonne necessarie
        self._verify_required_columns(data_to_use, soil_col)
        
        # Import solo qui per evitare importazioni circolari
        from core.interpolator import create_3d_interpolation as interp_func
        
        # Esegui l'interpolazione
        interp_result = interp_func(
            data_to_use,
            resolution=resolution,
            is_train_only=not use_test_data,
            soil_col=soil_col,
            method=method
        )
        
        # Salva l'interpolatore
        self.interpolator = interp_result['interpolator']
        
        return interp_result['grid_data']
    
    def _verify_required_columns(self, df, soil_col):
        """
        Verify that required columns are present in the dataframe
        
        Parameters:
        -----------
        df : DataFrame
            Dataframe to verify
        soil_col : str
            Soil column name
        """
        required_cols = ['x_coord', 'y_coord', df.columns[0]]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in data")
        
        # Stampa le prime righe per debug
        print(f"First 5 rows of data with {soil_col} column:")
        print(df[required_cols + [soil_col]].head())
    
    def create_3d_interpolation_alternative(self, resolution=10, use_test_data=False, soil_col='predicted_soil'):
        """
        Create a 3D interpolation of soil types using nearest neighbor method
        which is more robust than LinearNDInterpolator for problematic datasets
        
        Parameters:
        -----------
        resolution : int
            Resolution of the interpolation grid
        use_test_data : bool
            Whether to include test data in the interpolation
        soil_col : str
            Column name containing soil types to interpolate
            
        Returns:
        --------
        dict
            Grid data for 3D visualization
        """
        # Per compatibilità con il codice esistente, questo è ora solo un wrapper
        return self.create_3d_interpolation(
            resolution=resolution,
            use_test_data=use_test_data,
            soil_col=soil_col,
            method='nearest'
        )
    
    def visualize_comparative_models(self, resolution=10):
        """
        Crea e visualizza due modelli 3D del suolo:
        1. Un modello basato sulle predizioni del modello ML addestrato sui dati CPT
        2. Un modello geotecnico ottenuto dall'interpolazione diretta dei dati CPT reali
        
        Questa funzione è utile per confrontare le predizioni del modello ML con i dati reali
        e valutare l'accuratezza del modello ML in un contesto 3D.
        
        Parameters:
        -----------
        resolution : int
            Risoluzione della griglia di interpolazione (valore più piccolo = maggiore risoluzione)
                
        Returns:
        --------
        Figure object from plotting library
            Figura interattiva con i due modelli affiancati
        """
        if self.cpt_data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'soil []' not in self.cpt_data.columns:
            raise ValueError("No soil classification column found in the data")
        
        if 'predicted_soil' not in self.cpt_data.columns:
            raise ValueError("No predicted soil types. Call predict_soil_types() first.")
        
        print("\nCreazione di due modelli 3D per confronto:")
        print("1. Modello basato su predizioni ML")
        print("2. Modello geotecnico basato su dati CPT reali")
        
        # Create real soil model (geotechnical model)
        print("\nCreando modello geotecnico dai dati CPT reali...")
        real_model_data = self._create_soil_model('soil []', resolution, True)
        
        # Create ML soil model
        print("\nCreando modello basato sulle predizioni ML...")
        ml_model_data = self._create_soil_model('predicted_soil', resolution, True)
        
        # Visualize comparison
        if real_model_data and ml_model_data:
            print("\nVisualizzazione comparativa dei due modelli 3D...")
            return visualize_compare_3d_models(
                self.cpt_data,
                ml_model_data, 
                real_model_data,
                self.soil_types,
                self.soil_colors
            )
        return None
    
    def _create_soil_model(self, soil_col, resolution, use_test_data):
        """
        Create a soil model with a given soil column, trying different methods if necessary
        
        Parameters:
        -----------
        soil_col : str
            Soil column name
        resolution : int
            Resolution of the interpolation grid
        use_test_data : bool
            Whether to include test data in the interpolation
            
        Returns:
        --------
        dict or None
            Grid data for 3D visualization or None if failed
        """
        print(f"Creating soil model from '{soil_col}'...")
        try:
            # First try nearest neighbor method (more robust)
            return self.create_3d_interpolation(
                resolution=resolution,
                use_test_data=use_test_data,
                soil_col=soil_col,
                method='nearest'
            )
        except Exception as e:
            print(f"Nearest neighbor interpolation failed: {e}")
            print("Trying linear interpolation...")
            try:
                return self.create_3d_interpolation(
                    resolution=resolution,
                    use_test_data=use_test_data,
                    soil_col=soil_col,
                    method='linear'
                )
            except Exception as e:
                print(f"Failed to create soil model: {e}")
                return None
    
    def visualize_3d_model(self, interpolation_data=None, interactive=True, use_test_data=False):
        """
        Visualize a single 3D soil model
        
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
            interpolation_data = self._create_default_interpolation(use_test_data)
            if interpolation_data is None:
                return None
                    
        # Choose which dataset to visualize
        data_to_visualize = self.cpt_data if use_test_data else self.train_data
        
        return visualize_3d_model(
            data_to_visualize,
            interpolation_data,
            self.soil_types,
            self.soil_colors,
            interactive
        )
    
    def _create_default_interpolation(self, use_test_data):
        """
        Create default interpolation for visualization
        
        Parameters:
        -----------
        use_test_data : bool
            Whether to include test data in the interpolation
            
        Returns:
        --------
        dict or None
            Grid data for 3D visualization or None if failed
        """
        try:
            return self.create_3d_interpolation(
                use_test_data=use_test_data,
                method='nearest'
            )
        except Exception as e:
            print(f"Nearest neighbor interpolation failed: {e}")
            try:
                print("Trying linear interpolation...")
                return self.create_3d_interpolation(
                    use_test_data=use_test_data,
                    method='linear'
                )
            except Exception as e:
                print(f"Linear interpolation also failed: {e}")
                return None
    
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