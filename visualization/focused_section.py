import plotly.graph_objects as go
import numpy as np

def visualize_3d_model_section(cpt_data, interpolation_data, center_cpt=None, radius=10.0, 
                             soil_types=None, soil_colors=None):
    """
    Visualize a smaller section of the 3D soil model around specific CPT sites
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    interpolation_data : dict
        Output from create_3d_interpolation(), contains X, Y, Z, and values
    center_cpt : str, optional
        ID of the CPT to center the section around. If None, the middle of all CPTs is used
    radius : float
        Radius (in meters) of the section to visualize around the center point
    soil_types : list, optional
        List of soil types
    soil_colors : dict, optional
        Dictionary mapping soil types to colors
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive 3D visualization of the section
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
    
    # Get center coordinates
    if center_cpt is not None:
        # Use specified CPT as center
        center_data = cpt_data[cpt_data['cpt_id'] == center_cpt]
        if len(center_data) == 0:
            raise ValueError(f"CPT ID '{center_cpt}' not found in data")
        center_x = center_data['x_coord'].iloc[0]
        center_y = center_data['y_coord'].iloc[0]
    else:
        # Use center of all CPTs
        cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
        center_x = cpt_locations['x_coord'].mean()
        center_y = cpt_locations['y_coord'].mean()
    
    print(f"Center coordinates: X={center_x:.2f}, Y={center_y:.2f}")
    print(f"Section radius: {radius} meters")
    
    # Create mask for points within radius
    # For each point in the grid, calculate distance from center
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = distance <= radius
    
    # Apply mask to filter data
    X_section = X[mask]
    Y_section = Y[mask]
    Z_section = Z[mask]
    values_section = values[mask]
    
    # Create figure
    fig = go.Figure()
    
    # Create colormap and labels
    colorscale, tickvals, ticktext = _create_colormap_and_labels(soil_types, soil_colors)
    
    # Add a volume trace
    fig.add_trace(go.Volume(
        x=X_section,
        y=Y_section,
        z=Z_section,
        value=values_section,
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
    
    # Add CPT locations within the section
    cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
    cpt_distances = np.sqrt((cpt_locations['x_coord'] - center_x)**2 + 
                           (cpt_locations['y_coord'] - center_y)**2)
    section_cpts = cpt_locations[cpt_distances <= radius]
    
    # Get depth values
    depth_col = cpt_data.columns[0]
    z_min = Z.min()
    
    # Add CPT locations with different colors for training/testing if available
    if 'is_train' in cpt_data.columns:
        # Get training locations within section
        train_cpts = set(cpt_data[cpt_data['is_train'] == True]['cpt_id'].unique())
        train_section = section_cpts[section_cpts.index.isin(train_cpts)]
        
        # Get testing locations within section
        test_cpts = set(cpt_data[cpt_data['is_train'] == False]['cpt_id'].unique())
        test_section = section_cpts[section_cpts.index.isin(test_cpts)]
        
        # Add training CPTs
        if len(train_section) > 0:
            fig.add_trace(go.Scatter3d(
                x=train_section['x_coord'],
                y=train_section['y_coord'],
                z=[z_min for _ in range(len(train_section))],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='blue',
                    symbol='circle',
                    line=dict(color='black', width=1)
                ),
                text=train_section.index,
                textposition="top center",
                name='Training CPTs'
            ))
        
        # Add testing CPTs
        if len(test_section) > 0:
            fig.add_trace(go.Scatter3d(
                x=test_section['x_coord'],
                y=test_section['y_coord'],
                z=[z_min for _ in range(len(test_section))],
                mode='markers+text',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                text=test_section.index,
                textposition="top center",
                name='Testing CPTs'
            ))
    else:
        # Add all CPTs without distinction
        fig.add_trace(go.Scatter3d(
            x=section_cpts['x_coord'],
            y=section_cpts['y_coord'],
            z=[z_min for _ in range(len(section_cpts))],
            mode='markers+text',
            marker=dict(
                size=10,
                color='black',
                symbol='circle',
                line=dict(color='white', width=1)
            ),
            text=section_cpts.index,
            textposition="top center",
            name='CPT Locations'
        ))
    
    # Add a marker for the center point
    fig.add_trace(go.Scatter3d(
        x=[center_x],
        y=[center_y],
        z=[z_min],
        mode='markers',
        marker=dict(
            size=12,
            color='yellow',
            symbol='x',
            line=dict(color='black', width=2)
        ),
        name='Center Point'
    ))
    
    # Add a circle to show the section boundary
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = center_x + radius * np.cos(theta)
    circle_y = center_y + radius * np.sin(theta)
    
    fig.add_trace(go.Scatter3d(
        x=circle_x,
        y=circle_y,
        z=[z_min for _ in range(len(circle_x))],
        mode='lines',
        line=dict(color='green', width=3),
        name='Section Boundary'
    ))
    
    # Update layout
    fig.update_layout(
        title=f'3D Soil Model - {radius}m Section Around ({center_x:.1f}, {center_y:.1f})',
        scene=dict(
            xaxis_title='X Coordinate (m)',
            yaxis_title='Y Coordinate (m)',
            zaxis_title='Depth (m)',
            aspectratio=dict(x=1, y=1, z=0.7),  # Slight vertical exaggeration
            zaxis=dict(autorange='reversed')  # Invert Z axis for depth
        ),
        width=800,
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