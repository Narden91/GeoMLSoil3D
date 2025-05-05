import matplotlib.pyplot as plt
import numpy as np
from utils.soil_types import SoilTypeManager


def plot_cpt_locations(cpt_data, show_train_test=False):
    """
    Plot CPT test locations on a map, with optional train/test coloring
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    show_train_test : bool
        If True, color training and testing CPT locations differently
    """
    plt.figure(figsize=(10, 8))
    
    if show_train_test and 'is_train' in cpt_data.columns:
        _plot_cpt_locations_by_train_test(cpt_data)
    else:
        _plot_cpt_locations_simple(cpt_data)
    
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('CPT Test Locations')
    plt.grid(True)
    plt.show()


def _plot_cpt_locations_by_train_test(cpt_data):
    """
    Plot CPT locations with training/testing differentiation
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    """
    # Group by CPT ID and get the first row for each (they all have same coordinates)
    cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord', 'is_train']].first()
    
    # Separate training and testing locations
    train_locs = cpt_locations[cpt_locations['is_train'] == True]
    test_locs = cpt_locations[cpt_locations['is_train'] == False]
    
    # Plot training locations
    plt.scatter(train_locs['x_coord'], train_locs['y_coord'], 
                s=100, marker='^', color='blue', label='Training CPTs')
    
    # Plot testing locations
    plt.scatter(test_locs['x_coord'], test_locs['y_coord'], 
                s=100, marker='o', color='red', label='Testing CPTs')
    
    # Add labels
    for idx, row in train_locs.iterrows():
        plt.annotate(idx, (row['x_coord'], row['y_coord']), 
                    xytext=(5, 5), textcoords='offset points', color='blue')
    
    for idx, row in test_locs.iterrows():
        plt.annotate(idx, (row['x_coord'], row['y_coord']), 
                    xytext=(5, 5), textcoords='offset points', color='red')
    
    plt.legend()


def _plot_cpt_locations_simple(cpt_data):
    """
    Plot CPT locations without training/testing differentiation
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    """
    # Original behavior without train/test distinction
    cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
    plt.scatter(cpt_locations['x_coord'], cpt_locations['y_coord'], s=100, marker='^')
    
    for idx, row in cpt_locations.iterrows():
        plt.annotate(idx, (row['x_coord'], row['y_coord']), 
                    xytext=(5, 5), textcoords='offset points')


def plot_cpt_profile(cpt_data, example_cpt=None):
    """
    Plot CPT profile with cone resistance, friction ratio, and soil classification
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    example_cpt : str, optional
        ID of the CPT to plot, if None, uses the first one
    """
    if example_cpt is None and len(cpt_data['cpt_id'].unique()) > 0:
        example_cpt = cpt_data['cpt_id'].unique()[0]
    
    example_data = cpt_data[cpt_data['cpt_id'] == example_cpt]
    
    # Determine if we need to plot predicted soil
    include_predicted = 'predicted_soil' in example_data.columns
    
    # Create figure with appropriate number of subplots
    n_plots = 4 if (include_predicted and 'soil []' in example_data.columns) else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 10))
    
    # Depth column is typically the first column
    depth_col = cpt_data.columns[0]
    
    # Plot cone resistance
    axes[0].plot(example_data['qc [MPa]'], example_data[depth_col])
    axes[0].set_ylabel('Depth (m)')
    axes[0].set_xlabel('qc [MPa]')
    axes[0].set_title('Cone Resistance')
    axes[0].grid(True)
    axes[0].invert_yaxis()  # Depth increases downward
    
    # Plot friction ratio
    axes[1].plot(example_data['Rf [%]'], example_data[depth_col])
    axes[1].set_xlabel('Rf [%]')
    axes[1].set_title('Friction Ratio')
    axes[1].grid(True)
    axes[1].invert_yaxis()
    
    # Plot soil classification if available
    if 'soil []' in cpt_data.columns:
        _plot_soil_classification(axes[2], example_data, depth_col, 'soil []', 'Soil Classification')
        
        # If predicted soil is available, plot it alongside
        if include_predicted:
            _plot_soil_classification(axes[3], example_data, depth_col, 'predicted_soil', 'Model Prediction')
    
    plt.tight_layout()
    
    # Add a tag for training or testing data
    is_train = example_data['is_train'].iloc[0] if 'is_train' in example_data.columns else None
    dataset_type = "Training" if is_train else "Testing" if is_train is not None else ""
    plt.suptitle(f'CPT Profile: {example_cpt} ({dataset_type})', fontsize=16)
    
    plt.subplots_adjust(top=0.9)
    plt.show()


def _plot_soil_classification(ax, soil_data, depth_col, soil_col, title):
    """
    Plot soil classification on a given axis
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    soil_data : pandas.DataFrame
        DataFrame containing the soil data
    depth_col : str
        Name of the depth column
    soil_col : str
        Name of the soil classification column
    title : str
        Title for the plot
    """
    # Add some jitter to soil type for better visualization
    jitter = np.random.normal(0, 0.1, size=len(soil_data))
    
    # Creiamo una copia esplicita del DataFrame per evitare il SettingWithCopyWarning
    soil_data_copy = soil_data.copy()
    soil_data_copy['soil_jitter'] = soil_data_copy[soil_col] + jitter
    
    # Create a colormap with distinct colors for each soil type
    unique_soil_types = sorted(soil_data_copy[soil_col].unique())
    cmap = plt.cm.get_cmap('viridis', len(unique_soil_types))
    
    # Plot soil types with colors
    sc = ax.scatter(soil_data_copy['soil_jitter'], soil_data_copy[depth_col], 
                   c=soil_data_copy[soil_col], cmap=cmap, vmin=min(unique_soil_types)-0.5, 
                   vmax=max(unique_soil_types)+0.5)
    
    # Create custom tick labels with abbreviations
    tick_positions = np.arange(min(unique_soil_types), max(unique_soil_types)+1)
    tick_labels = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in tick_positions]
    
    # Add colorbar with custom ticks
    cbar = plt.colorbar(sc, ax=ax, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    
    ax.set_xlabel('Soil Type')
    ax.set_title(title)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.grid(True)
    ax.invert_yaxis()


def plot_feature_importance(model, feature_columns):
    """
    Plot feature importance from a trained model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_columns : list
        List of feature column names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importance for Soil Classification')
        plt.bar(range(len(feature_columns)), importances[indices], align='center')
        plt.xticks(range(len(feature_columns)), [feature_columns[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()


def plot_soil_legend():
    """
    Plot a legend with all soil types and their abbreviations
    """
    soil_types = SoilTypeManager.get_all_types()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, len(soil_types) * 0.4))
    
    # Create a simple legend/table
    for i, (soil_id, soil_info) in enumerate(soil_types.items()):
        abbr = soil_info['abbr']
        desc = soil_info['desc']
        
        ax.text(0.05, 1 - (i+1) * 0.1, f"{soil_id}", fontweight='bold')
        ax.text(0.15, 1 - (i+1) * 0.1, f"{abbr}", fontweight='bold')
        ax.text(0.25, 1 - (i+1) * 0.1, f"{desc}")
    
    ax.set_title("Soil Type Classification Legend")
    ax.axis('off')
    plt.tight_layout()
    plt.show()