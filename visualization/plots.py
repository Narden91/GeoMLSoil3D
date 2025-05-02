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
    else:
        # Original behavior without train/test distinction
        cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        plt.scatter(cpt_locations['x_coord'], cpt_locations['y_coord'], s=100, marker='^')
        
        for idx, row in cpt_locations.iterrows():
            plt.annotate(idx, (row['x_coord'], row['y_coord']), 
                        xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('CPT Test Locations')
    plt.grid(True)
    plt.show()


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
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    
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
        soil_data = example_data.copy()
        
        # Add some jitter to soil type for better visualization
        jitter = np.random.normal(0, 0.1, size=len(soil_data))
        soil_data['soil_jitter'] = soil_data['soil []'] + jitter
        
        # Create a colormap with distinct colors for each soil type
        unique_soil_types = sorted(soil_data['soil []'].unique())
        cmap = plt.cm.get_cmap('viridis', len(unique_soil_types))
        
        # Plot soil types with colors
        sc = axes[2].scatter(soil_data['soil_jitter'], soil_data[depth_col], 
                           c=soil_data['soil []'], cmap=cmap, vmin=min(unique_soil_types)-0.5, 
                           vmax=max(unique_soil_types)+0.5)
        
        # Create custom tick labels with abbreviations
        tick_positions = np.arange(min(unique_soil_types), max(unique_soil_types)+1)
        tick_labels = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in tick_positions]
        
        # Add colorbar with custom ticks
        cbar = plt.colorbar(sc, ax=axes[2], ticks=tick_positions)
        cbar.ax.set_yticklabels(tick_labels)
        
        axes[2].set_xlabel('Soil Type')
        axes[2].set_title('Soil Classification')
        axes[2].set_xticks(tick_positions)
        axes[2].set_xticklabels(tick_labels, rotation=45, ha='right')
        axes[2].grid(True)
        axes[2].invert_yaxis()
        
        # If predicted soil is available, plot it alongside
        if 'predicted_soil' in example_data.columns:
            # Create a fourth subplot for the predicted soil
            fig.set_size_inches(20, 10)  # Increase figure width
            ax_pred = fig.add_subplot(1, 4, 4)
            
            # Add jitter to predicted soil types
            soil_data['pred_jitter'] = soil_data['predicted_soil'] + jitter
            
            # Plot predicted soil types
            sc_pred = ax_pred.scatter(soil_data['pred_jitter'], soil_data[depth_col], 
                                    c=soil_data['predicted_soil'], cmap=cmap, 
                                    vmin=min(unique_soil_types)-0.5, 
                                    vmax=max(unique_soil_types)+0.5)
            
            # Add colorbar
            cbar_pred = plt.colorbar(sc_pred, ax=ax_pred, ticks=tick_positions)
            cbar_pred.ax.set_yticklabels(tick_labels)
            
            ax_pred.set_xlabel('Predicted Soil Type')
            ax_pred.set_title('Model Prediction')
            ax_pred.set_xticks(tick_positions)
            ax_pred.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax_pred.grid(True)
            ax_pred.invert_yaxis()
    
    plt.tight_layout()
    
    # Add a tag for training or testing data
    is_train = example_data['is_train'].iloc[0] if 'is_train' in example_data.columns else None
    dataset_type = "Training" if is_train else "Testing" if is_train is not None else ""
    plt.suptitle(f'CPT Profile: {example_cpt} ({dataset_type})', fontsize=16)
    
    plt.subplots_adjust(top=0.9)
    plt.show()


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
    
    # Corretto: usa plt.subplots() che restituisce sia figura che assi
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