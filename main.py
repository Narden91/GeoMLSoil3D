import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# Import local modules
from cpt_model import CPT_3D_SoilModel
from utils import debug_cpt_files
from soil_types import SoilTypeManager
from visualization import plot_soil_legend


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    else:
        print(f"Warning: Config file {config_path} not found. Using default settings.")
        return {}


def main(config_path="config.yaml"):
    """
    Main function to run the CPT 3D Soil Model framework
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Default values if not in config
    paths = config.get('paths', {})
    debug_cfg = config.get('debug', {})
    display = config.get('display', {})
    data_loading = config.get('data_loading', {})
    soil_classification = config.get('soil_classification', {})
    interpolation = config.get('interpolation', {})
    
    # Get paths from config
    data_path = paths.get('data', "data/CPT_*.csv")
    model_output = paths.get('model_output', "cpt_soil_model.pkl")
    
    # Create framework instance
    framework = CPT_3D_SoilModel()
    
    # Mostra la legenda dei tipi di suolo
    if display.get('show_soil_legend', True):
        print("Tipi di suolo e relative abbreviazioni:")
        soil_types = SoilTypeManager.get_all_types()
        for soil_id, soil_info in soil_types.items():
            print(f"{soil_id}: {soil_info['abbr']} - {soil_info['desc']}")
    
    # Analisi preliminare dei file CPT (opzionale)
    if debug_cfg.get('enabled', True):
        debug_cpt_files(data_path)
    
    # 1. Load data con coordinate generate in modo più robusto
    data = framework.load_data(
        data_path,
        x_coord_col=data_loading.get('x_coord_col'), 
        y_coord_col=data_loading.get('y_coord_col')
    )
    
    # Verifica che ci siano sufficienti coordinate uniche
    x_unique = framework.cpt_data['x_coord'].nunique()
    y_unique = framework.cpt_data['y_coord'].nunique()
    
    if debug_cfg.get('verbose', True):
        print(f"Dataset has {x_unique} unique x-coordinates and {y_unique} unique y-coordinates")
    
    if x_unique < 2 or y_unique < 2:
        print("WARNING: Not enough unique coordinates for 3D interpolation.")
        print("Consider adding proper spatial coordinates to your CPT files.")
    
    # 2. Explore data
    framework.explore_data(
        show_dataset_overview=display.get('show_dataset_overview', True),
        show_soil_distribution=display.get('show_soil_distribution', True),
        show_cpt_locations=display.get('plot_cpt_locations', True),
        show_cpt_profiles=display.get('plot_cpt_profiles', True)
    )
    
    # Visualizza la legenda dei tipi di suolo
    if display.get('show_soil_legend', True):
        plot_soil_legend()
    
    # 3. Train soil classification model
    if soil_classification.get('enabled', True):
        model_type = soil_classification.get('model_type', 'rf')
        test_size = soil_classification.get('test_size', 0.2)
        random_state = soil_classification.get('random_state', 42)
        
        model = framework.train_soil_classification_model(
            model_type=model_type,
            test_size=test_size,
            random_state=random_state
        )
    
    # 4. Predict soil types for all data points
    predictions = framework.predict_soil_types()
    
    # 5. Create 3D interpolation
    if interpolation.get('enabled', True):
        resolution = interpolation.get('resolution', 5)
        
        if interpolation.get('try_alternative_first', True):
            try:
                print("Trying alternative interpolation method...")
                interpolation_data = framework.create_3d_interpolation_alternative(resolution=resolution)
            except Exception as e:
                print(f"Alternative interpolation failed: {e}")
                print("Falling back to standard interpolation with robustness improvements...")
                try:
                    interpolation_data = framework.create_3d_interpolation(resolution=resolution)
                except Exception as e:
                    print(f"3D interpolation failed again: {e}")
                    print("Creating simplified 2D visualization instead...")
                    interpolation_data = None
        else:
            try:
                interpolation_data = framework.create_3d_interpolation(resolution=resolution)
            except Exception as e:
                print(f"3D interpolation failed: {e}")
                interpolation_data = None
    
        # 6. Visualize 3D model
        if interpolation_data is not None:
            try:
                framework.visualize_3d_model(
                    interpolation_data, 
                    interactive=display.get('interactive_visualization', True)
                )
            except Exception as e:
                print(f"3D visualization failed: {e}")
                print("Consider visualizing individual CPT profiles instead")
    
    # 7. Save model for future use
    framework.save_model(model_output)
    
    if display.get('show_soil_abbreviations', True):
        print("\nAbbreviazioni dei tipi di suolo:")
        for soil_id in sorted(framework.cpt_data['soil []'].unique()):
            abbr = SoilTypeManager.get_abbreviation(soil_id)
            desc = SoilTypeManager.get_description(soil_id)
            print(f"  {soil_id}: {abbr} - {desc}")


if __name__ == "__main__":
    import sys
    
    # Check if config path is provided as command line argument
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        main(config_path)
    else:
        main()