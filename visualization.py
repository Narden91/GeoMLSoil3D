import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D


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
        
        axes[2].scatter(soil_data['soil_jitter'], soil_data[depth_col], 
                       c=soil_data['soil []'], cmap='viridis')
        axes[2].set_xlabel('Soil Type')
        axes[2].set_title('Soil Classification')
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
        
        # Create colormap
        if soil_colors is None:
            # Default colormap
            colorscale = 'Viridis'
        else:
            # Custom colormap from soil types
            colorscale = []
            soil_types_sorted = sorted(soil_colors.keys())
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
                tickvals=soil_types if soil_types else None
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
        
        # Create scatter plot of soil types
        scatter = ax.scatter(
            X[::sample_step, ::sample_step, ::sample_step].flatten(),
            Y[::sample_step, ::sample_step, ::sample_step].flatten(),
            Z[::sample_step, ::sample_step, ::sample_step].flatten(),
            c=values[::sample_step, ::sample_step, ::sample_step].flatten(),
            cmap='viridis',
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
        
        # Add a color bar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Soil Type')
        
        plt.tight_layout()
        plt.show()
        
        return fig