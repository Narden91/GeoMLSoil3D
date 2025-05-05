import os
import sys
import traceback

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.model import CPT_3D_SoilModel
from utils.helpers import debug_cpt_files
from utils.soil_types import SoilTypeManager
from utils.config import (
    load_config,
    save_config,
    update_config_from_args,
    create_argument_parser
)
from visualization.plots import plot_soil_legend


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
    
    # Extract configuration sections
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
    
    # Show soil type legend if requested
    if display.get('show_soil_legend', True):
        _display_soil_legend()
    
    # Run preliminary debug on CPT files if enabled
    if debug_cfg.get('enabled', True):
        debug_cpt_files(data_path)
    
    # Load data and perform data exploration
    framework = _load_and_explore_data(framework, data_path, data_loading, display, 
                                        soil_classification, debug_cfg, paths)
    
    # Train and evaluate soil classification model if enabled
    if soil_classification.get('enabled', True):
        framework = _train_and_evaluate_model(framework, soil_classification, display)
    
    # Create and visualize 3D models if enabled
    if interpolation.get('enabled', True):
        _create_and_visualize_models(framework, interpolation, display)
    
    # Save model for future use
    framework.save_model(model_output)
    
    # Show soil abbreviations if requested
    if display.get('show_soil_abbreviations', True):
        _display_soil_abbreviations(framework)
    
    return framework


def _display_soil_legend():
    """Display soil types and their abbreviations"""
    print("Tipi di suolo e relative abbreviazioni:")
    soil_types = SoilTypeManager.get_all_types()
    for soil_id, soil_info in soil_types.items():
        print(f"{soil_id}: {soil_info['abbr']} - {soil_info['desc']}")
    
    # Visualize the legend
    plot_soil_legend()


def _load_and_explore_data(framework, data_path, data_loading, display, soil_classification, debug_cfg, paths):
    """
    Load data and perform data exploration
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        The soil model framework
    data_path : str
        Path to data files
    data_loading : dict
        Data loading configuration
    display : dict
        Display configuration
    soil_classification : dict
        Soil classification configuration
    debug_cfg : dict
        Debug configuration
    paths : dict
        Paths configuration
        
    Returns:
    --------
    CPT_3D_SoilModel
        Updated framework
    """
    # Load data with train/test split
    test_size = soil_classification.get('test_size', 0.2)
    random_state = soil_classification.get('random_state', 42)
    locations_path = paths.get('locations', 'location.csv')
    
    train_data, test_data = framework.load_data(
        data_path,
        x_coord_col=data_loading.get('x_coord_col'), 
        y_coord_col=data_loading.get('y_coord_col'),
        test_size=test_size,
        random_state=random_state,
        locations_path=locations_path
    )
    
    # Check for sufficient unique coordinates
    _check_coordinates(framework, debug_cfg)
    
    # Explore data
    framework.explore_data(
        show_dataset_overview=display.get('show_dataset_overview', True),
        show_soil_distribution=display.get('show_soil_distribution', True),
        show_cpt_locations=display.get('plot_cpt_locations', True),
        show_cpt_profiles=display.get('plot_cpt_profiles', True)
    )
    
    return framework


def _check_coordinates(framework, debug_cfg):
    """
    Check for sufficient unique coordinates
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        The soil model framework
    debug_cfg : dict
        Debug configuration
    """
    x_unique_train = framework.train_data['x_coord'].nunique()
    y_unique_train = framework.train_data['y_coord'].nunique()
    x_unique_test = framework.test_data['x_coord'].nunique()
    y_unique_test = framework.test_data['y_coord'].nunique()
    
    if debug_cfg.get('verbose', True):
        print(f"Training dataset has {x_unique_train} unique x-coordinates and {y_unique_train} unique y-coordinates")
        print(f"Testing dataset has {x_unique_test} unique x-coordinates and {y_unique_test} unique y-coordinates")
    
    if x_unique_train < 2 or y_unique_train < 2:
        print("WARNING: Not enough unique coordinates for 3D interpolation in training data.")
        print("Consider adding proper spatial coordinates to your CPT files.")


def _train_and_evaluate_model(framework, soil_classification, display):
    """
    Train and evaluate soil classification model
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        The soil model framework
    soil_classification : dict
        Soil classification configuration
    display : dict
        Display configuration
        
    Returns:
    --------
    CPT_3D_SoilModel
        Updated framework
    """
    model_type = soil_classification.get('model_type', 'rf')
    validation_size = 0.1  # Use a portion of training set for internal validation
    random_state = soil_classification.get('random_state', 42)
    
    # Train soil classification model
    model = framework.train_soil_classification_model(
        model_type=model_type,
        test_size=validation_size,
        random_state=random_state
    )
    
    # Evaluate on external test set
    test_metrics = framework.evaluate_on_test_data()
    
    # Predict soil types for all data points
    predictions = framework.predict_soil_types()
    
    # Display feature importance if requested
    if display.get('plot_feature_importance', True) and hasattr(model, 'feature_importances_'):
        from visualization.plots import plot_feature_importance
        plot_feature_importance(model, framework.feature_columns)
    
    return framework


def _create_and_visualize_models(framework, interpolation, display):
    """
    Create and visualize 3D models
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        The soil model framework
    interpolation : dict
        Interpolation configuration
    display : dict
        Display configuration
    """
    resolution = interpolation.get('resolution', 5)
    
    # Create and visualize comparative 3D models
    try:
        print("\nCreating comparative visualization of ML predictions vs Real CPT measurements...")
        framework.visualize_comparative_models(resolution=resolution)
    except Exception as e:
        print(f"Comparative 3D visualization failed: {e}")
        traceback.print_exc()
        
        print("\nFalling back to standard single model visualization...")
        # Create standard interpolation if comparative fails
        _create_standard_visualization(framework, interpolation, display)


def _create_standard_visualization(framework, interpolation, display):
    """
    Create and visualize standard 3D model
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        The soil model framework
    interpolation : dict
        Interpolation configuration
    display : dict
        Display configuration
    """
    resolution = interpolation.get('resolution', 5)
    try:
        method = 'nearest' if interpolation.get('try_alternative_first', True) else 'linear'
        interpolation_data = framework.create_3d_interpolation(
            resolution=resolution,
            use_test_data=True,
            method=method
        )
        
        # Visualize standard model
        framework.visualize_3d_model(
            interpolation_data=interpolation_data,
            interactive=display.get('interactive_visualization', True),
            use_test_data=True
        )
    except Exception as e:
        print(f"Standard 3D visualization also failed: {e}")
        print("Consider visualizing individual CPT profiles instead")


def _display_soil_abbreviations(framework):
    """
    Display soil abbreviations
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        The soil model framework
    """
    print("\nAbbreviazioni dei tipi di suolo:")
    for soil_id in sorted(framework.cpt_data['soil []'].unique()):
        abbr = SoilTypeManager.get_abbreviation(soil_id)
        desc = SoilTypeManager.get_description(soil_id)
        print(f"  {soil_id}: {abbr} - {desc}")
    
    # Print summary of model performance
    if hasattr(framework, 'test_metrics'):
        print("\nRiepilogo delle performance del modello:")
        print(f"Accuracy complessiva sul set di test: {framework.test_metrics['overall_accuracy']:.4f}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Load and update config with command line arguments
    config = load_config(args.config)
    config = update_config_from_args(config, args)
    
    # Save modified config
    save_config(config, 'config_modified.yaml')
    
    # Run with the modified config
    main('config_modified.yaml')