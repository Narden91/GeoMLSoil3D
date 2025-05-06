import os
import sys
import traceback

from core.model import CPT_3D_SoilModel
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
        
    Returns:
    --------
    CPT_3D_SoilModel
        Initialized and trained framework instance
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
    
    # Create framework instance
    framework = CPT_3D_SoilModel()
    
    # Show soil type legend if requested
    if display.get('show_soil_legend', True):
        plot_soil_legend()
    
    try:
        # Load and explore data
        framework = run_data_loading_phase(
            framework, 
            paths, 
            data_loading, 
            display, 
            soil_classification
        )
        
        # Train and evaluate soil classification model if enabled
        if soil_classification.get('enabled', True):
            framework = run_training_phase(
                framework, 
                soil_classification, 
                display
            )
        
        # Create and visualize 3D models if enabled
        if interpolation.get('enabled', True):
            run_visualization_phase(
                framework, 
                interpolation, 
                display
            )
        
        # Save model
        model_output = paths.get('model_output', "cpt_soil_model.pkl")
        framework.save_model(model_output)
        
        # Show soil abbreviations if requested
        if display.get('show_soil_abbreviations', True):
            display_soil_abbreviations(framework)
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        print("\nDespite the error, the framework object is being returned for manual inspection.")
    
    return framework


def run_data_loading_phase(framework, paths, data_loading, display, soil_classification):
    """
    Run data loading and exploration phase
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        Framework instance
    paths : dict
        Paths configuration
    data_loading : dict
        Data loading configuration
    display : dict
        Display configuration
    soil_classification : dict
        Soil classification configuration
        
    Returns:
    --------
    CPT_3D_SoilModel
        Updated framework instance
    """
    # Get data loading parameters
    data_path = paths.get('data', "data/CPT_*.csv")
    test_size = soil_classification.get('test_size', 0.2)
    random_state = soil_classification.get('random_state', 42)
    locations_path = paths.get('locations', 'location.csv')
    
    print(f"Loading data from: {data_path}")
    print(f"Test size: {test_size}, Random state: {random_state}")
    
    # Load data
    train_data, test_data = framework.load_data(
        data_path,
        x_coord_col=data_loading.get('x_coord_col'), 
        y_coord_col=data_loading.get('y_coord_col'),
        test_size=test_size,
        random_state=random_state,
        locations_path=locations_path
    )
    
    # Check dataset sizes
    print(f"Loaded {len(train_data)} training records and {len(test_data)} test records")
    check_data_quality(framework)
    
    # Explore data
    framework.explore_data(
        show_dataset_overview=display.get('show_dataset_overview', True),
        show_soil_distribution=display.get('show_soil_distribution', True),
        show_cpt_locations=display.get('plot_cpt_locations', True),
        show_cpt_profiles=display.get('plot_cpt_profiles', True)
    )
    
    return framework


def check_data_quality(framework):
    """
    Check data quality and provide warnings if needed
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        Framework with loaded data
    """
    # Check for sufficient unique coordinates
    x_unique_train = framework.train_data['x_coord'].nunique()
    y_unique_train = framework.train_data['y_coord'].nunique()
    x_unique_test = framework.test_data['x_coord'].nunique()
    y_unique_test = framework.test_data['y_coord'].nunique()
    
    print(f"Training dataset has {x_unique_train} unique x-coordinates and {y_unique_train} unique y-coordinates")
    print(f"Testing dataset has {x_unique_test} unique x-coordinates and {y_unique_test} unique y-coordinates")
    
    if x_unique_train < 2 or y_unique_train < 2:
        print("\nWARNING: Not enough unique coordinates for 3D interpolation in training data.")
        print("Consider adding proper spatial coordinates to your CPT files.")
    
    # Check if soil data is available
    if 'soil []' not in framework.cpt_data.columns:
        print("\nWARNING: No soil classification column found in data.")
        print("Soil classification needs to be computed from CPT parameters.")


def run_training_phase(framework, soil_classification, display):
    """
    Run model training and evaluation phase
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        Framework instance
    soil_classification : dict
        Soil classification configuration
    display : dict
        Display configuration
        
    Returns:
    --------
    CPT_3D_SoilModel
        Updated framework instance
    """
    model_type = soil_classification.get('model_type', 'rf')
    use_all_features = soil_classification.get('use_all_features', True)
    random_state = soil_classification.get('random_state', 42)
    
    print(f"\nTraining soil classification model...")
    print(f"Model type: {model_type}, Use all features: {use_all_features}")
    
    # Train model
    model = framework.train_soil_classification_model(
        model_type=model_type,
        test_size=0.1,  # Small internal validation split
        random_state=random_state,
        use_all_features=use_all_features
    )
    
    # Evaluate on separate test set
    print("\nEvaluating model on test data...")
    test_metrics = framework.evaluate_on_test_data()
    
    # Predict soil types for all data
    print("\nPredicting soil types for all data...")
    predictions = framework.predict_soil_types()
    
    return framework


def run_visualization_phase(framework, interpolation, display):
    """
    Run visualization phase for 3D models and cross-sections
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        Framework instance with trained model
    interpolation : dict
        Interpolation configuration
    display : dict
        Display configuration
    """
    resolution = interpolation.get('resolution', 5)
    
    print(f"\nCreating visualizations with resolution: {resolution}")
    
    # Create vertical CPT section if requested
    if display.get('cross_sections', False):
        try:
            print("\nCreating vertical CPT section...")
            framework.create_vertical_cpt_section(
                use_test_data=display.get('include_test_in_3d', False)
            )
        except Exception as e:
            print(f"Error creating vertical section: {e}")
    
    # Create interactive cross-section explorer if requested
    if display.get('cross_section_interactive', False):
        try:
            print("\nCreating interactive cross-section explorer...")
            framework.create_interactive_cross_section_explorer(
                use_test_data=display.get('include_test_in_3d', False)
            )
        except Exception as e:
            print(f"Error creating cross-section explorer: {e}")
    
    # Create comparative 3D models if requested
    if display.get('interactive_visualization', True):
        try:
            print("\nCreating comparative 3D visualization...")
            framework.visualize_comparative_models(resolution=resolution)
        except Exception as e:
            print(f"Error creating comparative 3D visualization: {e}")
            try_alternative_visualization(framework, interpolation, display)


def try_alternative_visualization(framework, interpolation, display):
    """
    Try alternative visualization methods if primary method fails
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        Framework instance
    interpolation : dict
        Interpolation configuration
    display : dict
        Display configuration
    """
    print("\nTrying alternative visualization method...")
    resolution = interpolation.get('resolution', 5)
    
    try:
        # Try using nearest neighbor interpolation
        interpolation_data = framework.create_3d_interpolation(
            resolution=resolution,
            use_test_data=True,
            method='nearest'
        )
        
        # Visualize model
        framework.visualize_3d_model(
            interpolation_data=interpolation_data,
            interactive=display.get('interactive_visualization', True),
            use_test_data=True
        )
    except Exception as e:
        print(f"Alternative visualization also failed: {e}")
        print("Consider visualizing only CPT profiles or cross-sections.")


def display_soil_abbreviations(framework):
    """
    Display soil abbreviations and model performance summary
    
    Parameters:
    -----------
    framework : CPT_3D_SoilModel
        Framework instance with trained model
    """
    print("\nSoil type abbreviations:")
    try:
        for soil_id in sorted(framework.cpt_data['soil []'].unique()):
            abbr = SoilTypeManager.get_abbreviation(soil_id)
            desc = SoilTypeManager.get_description(soil_id)
            print(f"  {soil_id}: {abbr} - {desc}")
        
        # Print summary of model performance if available
        if hasattr(framework, 'test_metrics'):
            print("\nModel performance summary:")
            print(f"Overall test accuracy: {framework.test_metrics['overall_accuracy']:.4f}")
    except Exception as e:
        print(f"Error displaying soil abbreviations: {e}")


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
    framework = main('config_modified.yaml')
    
    print("\nExecution completed. Framework object is available for inspection.")