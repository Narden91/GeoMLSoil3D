import plotly.graph_objects as go
import numpy as np

def visualize_3d_model_bounded(cpt_data, interpolation_data, 
                             soil_types=None, soil_colors=None, 
                             padding=5.0):
    """
    Visualize the 3D soil model constrained to the bounds of CPT locations
    
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
    padding : float
        Additional padding around the CPT bounds in meters
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D visualization
    """
    # Extract grid data
    if 'grid_data' in interpolation_data:
        X = interpolation_data['grid_data']['X']
        Y = interpolation_data['grid_data']['Y']
        Z = interpolation_data['grid_data']['Z']
        values = interpolation_data['grid_data']['values']
    else:
        X = interpolation_data['X']
        Y = interpolation_data['Y']
        Z = interpolation_data['Z']
        values = interpolation_data['values']
    
    # Get CPT locations
    cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
    
    # Calculate bounds with padding
    x_min = cpt_locations['x_coord'].min() - padding
    x_max = cpt_locations['x_coord'].max() + padding
    y_min = cpt_locations['y_coord'].min() - padding
    y_max = cpt_locations['y_coord'].max() + padding
    
    # Print bounds info
    print(f"CPT X range: {cpt_locations['x_coord'].min():.2f} to {cpt_locations['x_coord'].max():.2f}")
    print(f"CPT Y range: {cpt_locations['y_coord'].min():.2f} to {cpt_locations['y_coord'].max():.2f}")
    print(f"Visualization bounds with {padding}m padding:")
    print(f"X: {x_min:.2f} to {x_max:.2f}")
    print(f"Y: {y_min:.2f} to {y_max:.2f}")
    
    # Create mask for points within the bounds
    mask = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
    
    # Apply mask to filter data
    X_bounded = X[mask]
    Y_bounded = Y[mask]
    Z_bounded = Z[mask]
    values_bounded = values[mask]
    
    # Handle empty result (no points in bounds)
    if np.sum(mask) == 0:
        print("Warning: No points found within the bounds. Expanding bounds...")
        # Use full dataset instead
        X_bounded = X
        Y_bounded = Y
        Z_bounded = Z
        values_bounded = values
    else:
        print(f"Using {np.sum(mask)} points within the bounds out of {X.size} total points")
    
    # Create figure
    fig = go.Figure()
    
    # Create colormap and labels
    colorscale, tickvals, ticktext = _create_colormap_and_labels(soil_types, soil_colors)
    
    # Add a volume trace
    fig.add_trace(go.Volume(
        x=X_bounded,
        y=Y_bounded,
        z=Z_bounded,
        value=values_bounded,
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
    z_min = Z.min()
    
    # Add CPT locations with different colors for training/testing if available
    if 'is_train' in cpt_data.columns:
        # Get training locations
        train_cpts = set(cpt_data[cpt_data['is_train'] == True]['cpt_id'].unique())
        train_locs = cpt_locations[cpt_locations.index.isin(train_cpts)]
        
        # Get testing locations
        test_cpts = set(cpt_data[cpt_data['is_train'] == False]['cpt_id'].unique())
        test_locs = cpt_locations[cpt_locations.index.isin(test_cpts)] 
        
        # Add training CPTs
        if len(train_locs) > 0:
            fig.add_trace(go.Scatter3d(
                x=train_locs['x_coord'],
                y=train_locs['y_coord'],
                z=[z_min for _ in range(len(train_locs))],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='blue',
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                text=train_locs.index,
                textposition="top center",
                name='Training CPTs'
            ))
        
        # Add testing CPTs
        if len(test_locs) > 0:
            fig.add_trace(go.Scatter3d(
                x=test_locs['x_coord'],
                y=test_locs['y_coord'],
                z=[z_min for _ in range(len(test_locs))],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                text=test_locs.index,
                textposition="top center",
                name='Testing CPTs'
            ))
    else:
        # Add all CPTs without distinction
        fig.add_trace(go.Scatter3d(
            x=cpt_locations['x_coord'],
            y=cpt_locations['y_coord'],
            z=[z_min for _ in range(len(cpt_locations))],
            mode='markers+text',
            marker=dict(
                size=10,
                color='black',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            text=cpt_locations.index,
            textposition="top center",
            name='CPT Locations'
        ))
    
    # Add rectangle to show the bounded area
    rect_points = [
        [x_min, y_min], [x_max, y_min], 
        [x_max, y_max], [x_min, y_max], 
        [x_min, y_min]
    ]
    rect_x = [p[0] for p in rect_points]
    rect_y = [p[1] for p in rect_points]
    
    fig.add_trace(go.Scatter3d(
        x=rect_x,
        y=rect_y,
        z=[z_min for _ in range(len(rect_x))],
        mode='lines',
        line=dict(color='green', width=3),
        name='Bounds'
    ))
    
    # Update layout
    width_height_ratio = (x_max - x_min) / (y_max - y_min) if (y_max - y_min) > 0 else 1
    
    fig.update_layout(
        title=f'3D Soil Model - Bounded to CPT Locations (with {padding}m padding)',
        scene=dict(
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            zaxis_title='Depth (m)',
            aspectratio=dict(x=max(1, width_height_ratio), y=max(1, 1/width_height_ratio), z=0.7),
            aspectmode='manual',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max]),
            zaxis=dict(autorange='reversed')  # Invert Z axis for depth
        ),
        width=900,
        height=800,
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
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
        return 'Viridis', None, None
    
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
        from utils.soil_types import SoilTypeManager
        ticktext = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in soil_types_sorted]
        
        for i, soil_type in enumerate(soil_types_sorted):
            normalized_val = i / (len(soil_types_sorted) - 1)
            colorscale.append([normalized_val, soil_colors[soil_type]])
    
    return colorscale, tickvals, ticktext

