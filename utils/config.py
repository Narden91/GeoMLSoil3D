import yaml
import os
import argparse


def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            return config
    else:
        print(f"Warning: Config file {config_path} not found. Using default settings.")
        return {}


def save_config(config, output_path="config_modified.yaml"):
    """
    Save configuration to YAML file
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    output_path : str
        Path to save the configuration file
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {output_path}")


def update_config_from_args(config, args):
    """
    Update configuration with command line arguments
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    args : argparse.Namespace
        Command line arguments
        
    Returns:
    --------
    dict
        Updated configuration dictionary
    """
    # Override config with command line arguments
    if args.test_size is not None:
        if 'soil_classification' not in config:
            config['soil_classification'] = {}
        config['soil_classification']['test_size'] = args.test_size
    
    if args.model_type is not None:
        if 'soil_classification' not in config:
            config['soil_classification'] = {}
        config['soil_classification']['model_type'] = args.model_type
    
    if args.use_test_in_3d:
        if 'display' not in config:
            config['display'] = {}
        config['display']['include_test_in_3d'] = True
    
    return config


def create_argument_parser():
    """
    Create argument parser for command line interface
    
    Returns:
    --------
    argparse.ArgumentParser
        Argument parser
    """
    parser = argparse.ArgumentParser(description="CPT 3D Soil Modeling Framework")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--test-size', type=float, help='Proportion of CPT files to use for testing')
    parser.add_argument('--model-type', type=str, choices=['rf', 'xgb'], help='Machine learning model type')
    parser.add_argument('--use-test-in-3d', action='store_true', help='Include test data in 3D visualization')
    
    return parser


def get_default_config():
    """
    Get default configuration
    
    Returns:
    --------
    dict
        Default configuration dictionary
    """
    return {
        "paths": {
            "data": "data/CPT_*.csv",
            "model_output": "cpt_soil_model.pkl",
            "output_dir": "output/"
        },
        "debug": {
            "enabled": False,
            "verbose": False,
            "save_logs": False,
            "log_file": "geoml_soil_3d.log"
        },
        "display": {
            "show_soil_legend": False,
            "show_soil_abbreviations": False,
            "show_dataset_overview": True,
            "show_soil_distribution": True,
            "plot_cpt_locations": True,
            "plot_cpt_profiles": True,
            "plot_feature_importance": True,
            "interactive_visualization": True,
            "cross_sections": False,
            "include_test_in_3d": False
        },
        "data_loading": {
            "add_artificial_coordinates": True,
            "coordinate_jitter": 0.5,
            "spiral_pattern": True,
            "x_coord_col": None,
            "y_coord_col": None
        },
        "soil_classification": {
            "enabled": True,
            "model_type": "rf",
            "test_size": 0.2,
            "random_state": 42,
            "compute_missing": True,
            "features": [
                "qc [MPa]",
                "fs [MPa]",
                "Rf [%]",
                "u2 [MPa]"
            ]
        },
        "interpolation": {
            "enabled": True,
            "resolution": 5,
            "try_alternative_first": True,
            "margin": 5,
            "z_resolution_factor": 0.5,
            "vertical_exaggeration": 0.5
        },
        "styling": {
            "colormap": "viridis",
            "point_size": 5,
            "line_width": 1.5,
            "soil_colors": {}
        }
    }