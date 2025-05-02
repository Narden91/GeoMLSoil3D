import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from soil_types import SoilTypeManager


def plot_cpt_locations(cpt_data):
    """
    Plot CPT test locations on a map
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    """
    plt.figure(figsize=(10, 8))
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
    
    plt.tight_layout()
    plt.suptitle(f'CPT Profile: {example_cpt}', fontsize=16)
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


def visualize_3d_model(cpt_data, interpolation_data, soil_types=None, soil_colors=None, interactive=True):
    """
    Visualize the 3D soil model using either plotly or matplotlib
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    interpolation_data : dict
        Output from create_3d_interpolation(), contains X, Y, Z, and values
    soil_types : list, optional
        List of soil types
    soil_colors : dict, optional
        Dictionary mapping soil types to colors
    interactive : bool
        Whether to create an interactive Plotly visualization
    """
    X = interpolation_data['X']
    Y = interpolation_data['Y']
    Z = interpolation_data['Z']
    values = interpolation_data['values']
    
    print("Creating 3D visualization...")
    
    if interactive:
        # Create interactive 3D visualization with Plotly
        fig = go.Figure()
        
        # Create colormap and labels
        if soil_colors is None:
            # Default colormap
            colorscale = 'Viridis'
            tickvals = None
            ticktext = None
        else:
            # Custom colormap from soil types
            colorscale = []
            soil_types_sorted = sorted(soil_colors.keys())
            tickvals = soil_types_sorted
            
            # Create tick labels with abbreviations
            if isinstance(soil_colors[soil_types_sorted[0]], dict):
                # Nuovo formato con label
                ticktext = [soil_colors[soil_type]['label'] for soil_type in soil_types_sorted]
                
                for i, soil_type in enumerate(soil_types_sorted):
                    normalized_val = i / (len(soil_types_sorted) - 1)
                    colorscale.append([normalized_val, soil_colors[soil_type]['color']])
            else:
                # Vecchio formato
                ticktext = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in soil_types_sorted]
                
                for i, soil_type in enumerate(soil_types_sorted):
                    normalized_val = i / (len(soil_types_sorted) - 1)
                    colorscale.append([normalized_val, soil_colors[soil_type]])
        
        # Add a volume trace
        fig.add_trace(go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=values.flatten(),
            isomin=min(soil_types) if soil_types else 0,
            isomax=max(soil_types) if soil_types else 10,
            opacity=0.6,
            surface_count=15,  # Controls the level of detail
            colorscale=colorscale,
            colorbar=dict(
                title='Soil Type',
                tickvals=tickvals,
                ticktext=ticktext
            )
        ))
        
        # Add CPT locations for reference
        cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        fig.add_trace(go.Scatter3d(
            x=cpt_locations['x_coord'],
            y=cpt_locations['y_coord'],
            z=[Z.min() for _ in range(len(cpt_locations))],  # Place at surface
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol='circle'
            ),
            text=cpt_locations.index,
            name='CPT Locations'
        ))
        
        # Update layout
        fig.update_layout(
            title='3D Soil Composition Model',
            scene=dict(
                xaxis_title='X Coordinate (m)',
                yaxis_title='Y Coordinate (m)',
                zaxis_title='Depth (m)',
                aspectratio=dict(x=1, y=1, z=0.5),  # Vertical exaggeration
                zaxis=dict(autorange='reversed')  # Invert Z axis for depth
            ),
            width=900,
            height=800,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        fig.show()
        return fig
    
    else:
        # Create static 3D visualization with matplotlib
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # We'll use a scatter plot with points colored by soil type
        # Sampling the grid to avoid too many points
        sample_step = 3
        
        # Prepare colormap
        unique_soil_types = sorted(np.unique(values))
        cmap = plt.cm.get_cmap('viridis', len(unique_soil_types))
        
        # Create scatter plot of soil types
        scatter = ax.scatter(
            X[::sample_step, ::sample_step, ::sample_step].flatten(),
            Y[::sample_step, ::sample_step, ::sample_step].flatten(),
            Z[::sample_step, ::sample_step, ::sample_step].flatten(),
            c=values[::sample_step, ::sample_step, ::sample_step].flatten(),
            cmap=cmap,
            alpha=0.3,
            marker='o',
            s=5
        )
        
        # Add CPT locations
        cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        ax.scatter(
            cpt_locations['x_coord'],
            cpt_locations['y_coord'],
            [Z.min() for _ in range(len(cpt_locations))],
            color='black',
            marker='^',
            s=100,
            label='CPT Locations'
        )
        
        # Set labels and title
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_zlabel('Depth (m)')
        ax.set_title('3D Soil Composition Model')
        
        # Invert z-axis for depth
        ax.invert_zaxis()
        
        # Add a color bar with custom tick labels
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1, ticks=unique_soil_types)
        cbar.set_label('Soil Type')
        
        # Set tick labels with abbreviations
        tick_labels = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in unique_soil_types]
        cbar.ax.set_yticklabels(tick_labels)
        
        plt.tight_layout()
        plt.show()
        
        return fig


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