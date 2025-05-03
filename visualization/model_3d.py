import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from utils.soil_types import SoilTypeManager


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
    
    # Check if we're visualizing training or all data
    is_train_only = 'is_train' in cpt_data.columns and all(cpt_data['is_train'])
    dataset_type = "Training Data Only" if is_train_only else "All Data (Training + Testing)"
    
    if interactive:
        return _create_interactive_3d_visualization(
            cpt_data, X, Y, Z, values, soil_types, soil_colors, dataset_type
        )
    else:
        return _create_static_3d_visualization(
            cpt_data, X, Y, Z, values, soil_types, soil_colors, dataset_type
        )


def _create_interactive_3d_visualization(cpt_data, X, Y, Z, values, soil_types, soil_colors, dataset_type):
    """
    Create interactive 3D visualization with Plotly
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    X, Y, Z : ndarray
        Meshgrid coordinates
    values : ndarray
        Soil type values
    soil_types : list
        List of soil types
    soil_colors : dict
        Dictionary mapping soil types to colors
    dataset_type : str
        Type of dataset being visualized
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D visualization
    """
    # Create figure
    fig = go.Figure()
    
    # Create colormap and labels
    colorscale, tickvals, ticktext = _create_colormap_and_labels(soil_types, soil_colors)
    
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
    
    # Add CPT locations
    _add_cpt_locations_to_figure(fig, cpt_data, Z.min())
    
    # Update layout
    fig.update_layout(
        title=f'3D Soil Composition Model ({dataset_type})',
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


def _create_colormap_and_labels(soil_types, soil_colors):
    """
    Create colormap and labels for 3D visualization
    
    Parameters:
    -----------
    soil_types : list
        List of soil types
    soil_colors : dict
        Dictionary mapping soil types to colors
        
    Returns:
    --------
    colorscale, tickvals, ticktext : tuple
        Colorscale, tick values, and tick text for visualization
    """
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
            # New format with label
            ticktext = [soil_colors[soil_type]['label'] for soil_type in soil_types_sorted]
            
            for i, soil_type in enumerate(soil_types_sorted):
                normalized_val = i / (len(soil_types_sorted) - 1)
                colorscale.append([normalized_val, soil_colors[soil_type]['color']])
        else:
            # Old format
            ticktext = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in soil_types_sorted]
            
            for i, soil_type in enumerate(soil_types_sorted):
                normalized_val = i / (len(soil_types_sorted) - 1)
                colorscale.append([normalized_val, soil_colors[soil_type]])
    
    return colorscale, tickvals, ticktext


def _add_cpt_locations_to_figure(fig, cpt_data, z_min):
    """
    Add CPT locations to 3D figure
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to add locations to
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    z_min : float
        Minimum Z value for placing CPT markers
    """
    if 'is_train' in cpt_data.columns:
        # Get training locations
        train_locs = cpt_data[cpt_data['is_train'] == True].groupby('cpt_id')[['x_coord', 'y_coord']].first()
        if len(train_locs) > 0:
            fig.add_trace(go.Scatter3d(
                x=train_locs['x_coord'],
                y=train_locs['y_coord'],
                z=[z_min for _ in range(len(train_locs))],  # Place at surface
                mode='markers',
                marker=dict(
                    size=8,
                    color='blue',
                    symbol='circle'
                ),
                text=train_locs.index,
                name='Training CPT Locations'
            ))
        
        # Get testing locations
        test_locs = cpt_data[cpt_data['is_train'] == False].groupby('cpt_id')[['x_coord', 'y_coord']].first()
        if len(test_locs) > 0:
            fig.add_trace(go.Scatter3d(
                x=test_locs['x_coord'],
                y=test_locs['y_coord'],
                z=[z_min for _ in range(len(test_locs))],  # Place at surface
                mode='markers',
                marker=dict(
                    size=8,
                    color='red',
                    symbol='diamond'
                ),
                text=test_locs.index,
                name='Testing CPT Locations'
            ))
    else:
        # Original behavior without train/test distinction
        cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        fig.add_trace(go.Scatter3d(
            x=cpt_locations['x_coord'],
            y=cpt_locations['y_coord'],
            z=[z_min for _ in range(len(cpt_locations))],  # Place at surface
            mode='markers',
            marker=dict(
                size=8,
                color='black',
                symbol='circle'
            ),
            text=cpt_locations.index,
            name='CPT Locations'
        ))


def _create_static_3d_visualization(cpt_data, X, Y, Z, values, soil_types, soil_colors, dataset_type):
    """
    Create static 3D visualization with matplotlib
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    X, Y, Z : ndarray
        Meshgrid coordinates
    values : ndarray
        Soil type values
    soil_types : list
        List of soil types
    soil_colors : dict
        Dictionary mapping soil types to colors
    dataset_type : str
        Type of dataset being visualized
        
    Returns:
    --------
    matplotlib.figure.Figure
        Static 3D visualization
    """
    # Create figure
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
    _add_cpt_locations_to_matplotlib(ax, cpt_data, Z.min())
    
    # Set labels and title
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')
    ax.set_zlabel('Depth (m)')
    ax.set_title(f'3D Soil Composition Model ({dataset_type})')
    
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


def _add_cpt_locations_to_matplotlib(ax, cpt_data, z_min):
    """
    Add CPT locations to matplotlib 3D axis
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axis to add locations to
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    z_min : float
        Minimum Z value for placing CPT markers
    """
    if 'is_train' in cpt_data.columns:
        # Get training locations
        train_locs = cpt_data[cpt_data['is_train'] == True].groupby('cpt_id')[['x_coord', 'y_coord']].first()
        if len(train_locs) > 0:
            ax.scatter(
                train_locs['x_coord'],
                train_locs['y_coord'],
                [z_min for _ in range(len(train_locs))],
                color='blue',
                marker='^',
                s=100,
                label='Training CPTs'
            )
        
        # Get testing locations
        test_locs = cpt_data[cpt_data['is_train'] == False].groupby('cpt_id')[['x_coord', 'y_coord']].first()
        if len(test_locs) > 0:
            ax.scatter(
                test_locs['x_coord'],
                test_locs['y_coord'],
                [z_min for _ in range(len(test_locs))],
                color='red',
                marker='o',
                s=100,
                label='Testing CPTs'
            )
        
        ax.legend()
    else:
        # Original behavior without train/test distinction
        cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        ax.scatter(
            cpt_locations['x_coord'],
            cpt_locations['y_coord'],
            [z_min for _ in range(len(cpt_locations))],
            color='black',
            marker='^',
            s=100,
            label='CPT Locations'
        )


def visualize_compare_3d_models(cpt_data, ml_model_data, real_model_data, 
                              soil_types=None, soil_colors=None):
    """
    Visualize comparison between ML predicted and real CPT 3D soil models
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    ml_model_data : dict
        Interpolation from ML predicted soil types
    real_model_data : dict
        Interpolation from actual CPT measurements
    soil_types : list, optional
        List of soil types
    soil_colors : dict, optional
        Dictionary mapping soil types to colors
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive comparison visualization
    """
    # Extract data from both models
    X_ml = ml_model_data['X']
    Y_ml = ml_model_data['Y']
    Z_ml = ml_model_data['Z']
    values_ml = ml_model_data['values']
    
    X_real = real_model_data['X']
    Y_real = real_model_data['Y']
    Z_real = real_model_data['Z']
    values_real = real_model_data['values']
    
    print("Creating comparative 3D visualization...")
    
    # Create layout optimized for two 3D plots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=('ML Predicted Soil Model', 'Actual CPT Measurements Model'),
        horizontal_spacing=0.05  # Reduce spacing between plots
    )
    
    # Create colormap and labels
    colorscale, tickvals, ticktext = _create_colormap_and_labels(soil_types, soil_colors)
    
    # Add ML model to first subplot
    _add_model_to_subplot(
        fig, X_ml, Y_ml, Z_ml, values_ml, soil_types, 
        colorscale, tickvals, ticktext, 0.45, 'ML Predicted Soil',
        1, 1
    )
    
    # Add real model to second subplot
    _add_model_to_subplot(
        fig, X_real, Y_real, Z_real, values_real, soil_types, 
        colorscale, tickvals, ticktext, 0.95, 'Actual CPT Measurements',
        1, 2
    )
    
    # Add CPT locations to both subplots
    _add_cpt_locations_to_comparative_figure(fig, cpt_data, Z_ml.min(), Z_real.min())
    
    # Update both scenes to match dimensions and orientation
    for i in [1, 2]:
        fig.update_scenes(
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            zaxis_title='Depth (m)',
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),  # Vertical exaggeration
            zaxis=dict(autorange='reversed'),  # Invert Z axis for depth
            row=1, col=i
        )
    
    # Update overall layout
    fig.update_layout(
        title='Comparison: ML Predictions vs. Actual CPT Measurements',
        width=1600,  # Wider display
        height=800,
        margin=dict(l=20, r=20, b=20, t=50)  # Minimal margins
    )
    
    fig.show()
    return fig


def _add_model_to_subplot(fig, X, Y, Z, values, soil_types, colorscale, tickvals, ticktext, 
                         colorbar_x, name, row, col):
    """
    Add a 3D soil model to a subplot
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to add model to
    X, Y, Z : ndarray
        Meshgrid coordinates
    values : ndarray
        Soil type values
    soil_types : list
        List of soil types
    colorscale : list
        Colorscale for visualization
    tickvals : list
        Tick values for colorbar
    ticktext : list
        Tick text for colorbar
    colorbar_x : float
        X position of colorbar
    name : str
        Name of the model
    row, col : int
        Row and column of subplot
    """
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
            ticktext=ticktext,
            x=colorbar_x  # Position colorbar
        ),
        name=name
    ), row=row, col=col)


def _add_cpt_locations_to_comparative_figure(fig, cpt_data, z_min_ml, z_min_real):
    """
    Add CPT locations to both subplots in comparative figure
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Figure to add locations to
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    z_min_ml : float
        Minimum Z value for ML model subplot
    z_min_real : float
        Minimum Z value for real model subplot
    """
    # Get CPT locations
    cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
    
    # Add to ML model subplot
    fig.add_trace(go.Scatter3d(
        x=cpt_locations['x_coord'],
        y=cpt_locations['y_coord'],
        z=[z_min_ml for _ in range(len(cpt_locations))],  # Place at surface
        mode='markers',
        marker=dict(
            size=8,
            color='black',
            symbol='circle'
        ),
        text=cpt_locations.index,
        name='CPT Locations'
    ), row=1, col=1)
    
    # Add to real model subplot
    fig.add_trace(go.Scatter3d(
        x=cpt_locations['x_coord'],
        y=cpt_locations['y_coord'],
        z=[z_min_real for _ in range(len(cpt_locations))],  # Place at surface
        mode='markers',
        marker=dict(
            size=8,
            color='black',
            symbol='circle'
        ),
        text=cpt_locations.index,
        name='CPT Locations',
        showlegend=False  # Don't show duplicate legend entry
    ), row=1, col=2)