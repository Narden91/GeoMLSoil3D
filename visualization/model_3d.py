import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
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
        
        # Add CPT locations for reference with different colors for train/test
        if 'is_train' in cpt_data.columns:
            # Get training locations
            train_locs = cpt_data[cpt_data['is_train'] == True].groupby('cpt_id')[['x_coord', 'y_coord']].first()
            if len(train_locs) > 0:
                fig.add_trace(go.Scatter3d(
                    x=train_locs['x_coord'],
                    y=train_locs['y_coord'],
                    z=[Z.min() for _ in range(len(train_locs))],  # Place at surface
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
                    z=[Z.min() for _ in range(len(test_locs))],  # Place at surface
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
        
        # Add CPT locations with different colors for train/test
        if 'is_train' in cpt_data.columns:
            # Get training locations
            train_locs = cpt_data[cpt_data['is_train'] == True].groupby('cpt_id')[['x_coord', 'y_coord']].first()
            if len(train_locs) > 0:
                ax.scatter(
                    train_locs['x_coord'],
                    train_locs['y_coord'],
                    [Z.min() for _ in range(len(train_locs))],
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
                    [Z.min() for _ in range(len(test_locs))],
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
    

def visualize_compare_3d_models(cpt_data, pred_interpolation_data, real_interpolation_data, 
                              soil_types=None, soil_colors=None, interactive=True):
    """
    Visualize comparison between predicted and real 3D soil models
    
    Parameters:
    -----------
    cpt_data : pandas.DataFrame
        DataFrame containing the CPT data
    pred_interpolation_data : dict
        Output from create_3d_interpolation() for predicted soil
    real_interpolation_data : dict
        Output from create_3d_interpolation() for real soil
    soil_types : list, optional
        List of soil types
    soil_colors : dict, optional
        Dictionary mapping soil types to colors
    interactive : bool
        Whether to create an interactive Plotly visualization
    """
    # Estrai i dati dai due modelli
    X_pred = pred_interpolation_data['X']
    Y_pred = pred_interpolation_data['Y']
    Z_pred = pred_interpolation_data['Z']
    values_pred = pred_interpolation_data['values']
    
    X_real = real_interpolation_data['X']
    Y_real = real_interpolation_data['Y']
    Z_real = real_interpolation_data['Z']
    values_real = real_interpolation_data['values']
    
    print("Creating comparative 3D visualization...")
    
    # Check if we're visualizing training or all data
    is_train_only = 'is_train' in cpt_data.columns and all(cpt_data['is_train'])
    dataset_type = "Training Data Only" if is_train_only else "All Data (Training + Testing)"
    
    if interactive:
        # Create interactive 3D visualization with Plotly
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=('Predicted Soil Model', 'Real Soil Model')
        )
        
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
        
        # Add volume trace for predicted model
        fig.add_trace(go.Volume(
            x=X_pred.flatten(),
            y=Y_pred.flatten(),
            z=Z_pred.flatten(),
            value=values_pred.flatten(),
            isomin=min(soil_types) if soil_types else 0,
            isomax=max(soil_types) if soil_types else 10,
            opacity=0.6,
            surface_count=15,  # Controls the level of detail
            colorscale=colorscale,
            colorbar=dict(
                title='Soil Type',
                tickvals=tickvals,
                ticktext=ticktext,
                x=0.45  # Position colorbar for first subplot
            ),
            name='Predicted Soil'
        ), row=1, col=1)
        
        # Add volume trace for real model
        fig.add_trace(go.Volume(
            x=X_real.flatten(),
            y=Y_real.flatten(),
            z=Z_real.flatten(),
            value=values_real.flatten(),
            isomin=min(soil_types) if soil_types else 0,
            isomax=max(soil_types) if soil_types else 10,
            opacity=0.6,
            surface_count=15,  # Controls the level of detail
            colorscale=colorscale,
            colorbar=dict(
                title='Soil Type',
                tickvals=tickvals,
                ticktext=ticktext,
                x=1.0  # Position colorbar for second subplot
            ),
            name='Real Soil'
        ), row=1, col=2)
        
        # Add CPT locations for reference with different colors for train/test
        if 'is_train' in cpt_data.columns:
            # Get training locations
            train_locs = cpt_data[cpt_data['is_train'] == True].groupby('cpt_id')[['x_coord', 'y_coord']].first()
            if len(train_locs) > 0:
                # Add to predicted model subplot
                fig.add_trace(go.Scatter3d(
                    x=train_locs['x_coord'],
                    y=train_locs['y_coord'],
                    z=[Z_pred.min() for _ in range(len(train_locs))],  # Place at surface
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        symbol='circle'
                    ),
                    text=train_locs.index,
                    name='Training CPT Locations'
                ), row=1, col=1)
                
                # Add to real model subplot
                fig.add_trace(go.Scatter3d(
                    x=train_locs['x_coord'],
                    y=train_locs['y_coord'],
                    z=[Z_real.min() for _ in range(len(train_locs))],  # Place at surface
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='blue',
                        symbol='circle'
                    ),
                    text=train_locs.index,
                    name='Training CPT Locations',
                    showlegend=False  # Don't show duplicate legend entry
                ), row=1, col=2)
            
            # Get testing locations
            test_locs = cpt_data[cpt_data['is_train'] == False].groupby('cpt_id')[['x_coord', 'y_coord']].first()
            if len(test_locs) > 0:
                # Add to predicted model subplot
                fig.add_trace(go.Scatter3d(
                    x=test_locs['x_coord'],
                    y=test_locs['y_coord'],
                    z=[Z_pred.min() for _ in range(len(test_locs))],  # Place at surface
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond'
                    ),
                    text=test_locs.index,
                    name='Testing CPT Locations'
                ), row=1, col=1)
                
                # Add to real model subplot
                fig.add_trace(go.Scatter3d(
                    x=test_locs['x_coord'],
                    y=test_locs['y_coord'],
                    z=[Z_real.min() for _ in range(len(test_locs))],  # Place at surface
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond'
                    ),
                    text=test_locs.index,
                    name='Testing CPT Locations',
                    showlegend=False  # Don't show duplicate legend entry
                ), row=1, col=2)
        else:
            # Original behavior without train/test distinction
            cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
            
            # Add to predicted model subplot
            fig.add_trace(go.Scatter3d(
                x=cpt_locations['x_coord'],
                y=cpt_locations['y_coord'],
                z=[Z_pred.min() for _ in range(len(cpt_locations))],  # Place at surface
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
                z=[Z_real.min() for _ in range(len(cpt_locations))],  # Place at surface
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
        
        # Update layout for both subplots
        for i in [1, 2]:
            fig.update_scenes(
                xaxis_title='X Coordinate (m)',
                yaxis_title='Y Coordinate (m)',
                zaxis_title='Depth (m)',
                aspectratio=dict(x=1, y=1, z=0.5),  # Vertical exaggeration
                zaxis=dict(autorange='reversed'),  # Invert Z axis for depth
                row=1, col=i
            )
        
        # Update overall layout
        fig.update_layout(
            title=f'Comparison of Predicted vs Real 3D Soil Models ({dataset_type})',
            width=1200,
            height=800,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        fig.show()
        return fig
    
    else:
        # Create static 3D visualization with matplotlib
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        from utils.soil_types import SoilTypeManager
        
        fig = plt.figure(figsize=(16, 8))
        
        # Predicted soil model
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Real soil model
        ax2 = fig.add_subplot(122, projection='3d')
        
        # We'll use scatter plots with points colored by soil type
        # Sampling the grid to avoid too many points
        sample_step = 3
        
        # Prepare colormap
        unique_soil_types = sorted(np.unique(np.concatenate([values_pred.flatten(), values_real.flatten()])))
        cmap = plt.cm.get_cmap('viridis', len(unique_soil_types))
        
        # Create scatter plot for predicted soil types
        scatter1 = ax1.scatter(
            X_pred[::sample_step, ::sample_step, ::sample_step].flatten(),
            Y_pred[::sample_step, ::sample_step, ::sample_step].flatten(),
            Z_pred[::sample_step, ::sample_step, ::sample_step].flatten(),
            c=values_pred[::sample_step, ::sample_step, ::sample_step].flatten(),
            cmap=cmap,
            alpha=0.3,
            marker='o',
            s=5
        )
        
        # Create scatter plot for real soil types
        scatter2 = ax2.scatter(
            X_real[::sample_step, ::sample_step, ::sample_step].flatten(),
            Y_real[::sample_step, ::sample_step, ::sample_step].flatten(),
            Z_real[::sample_step, ::sample_step, ::sample_step].flatten(),
            c=values_real[::sample_step, ::sample_step, ::sample_step].flatten(),
            cmap=cmap,
            alpha=0.3,
            marker='o',
            s=5
        )
        
        # Add CPT locations with different colors for train/test
        if 'is_train' in cpt_data.columns:
            # Get training locations
            train_locs = cpt_data[cpt_data['is_train'] == True].groupby('cpt_id')[['x_coord', 'y_coord']].first()
            if len(train_locs) > 0:
                ax1.scatter(
                    train_locs['x_coord'],
                    train_locs['y_coord'],
                    [Z_pred.min() for _ in range(len(train_locs))],
                    color='blue',
                    marker='^',
                    s=100,
                    label='Training CPTs'
                )
                
                ax2.scatter(
                    train_locs['x_coord'],
                    train_locs['y_coord'],
                    [Z_real.min() for _ in range(len(train_locs))],
                    color='blue',
                    marker='^',
                    s=100,
                    label='Training CPTs'
                )
            
            # Get testing locations
            test_locs = cpt_data[cpt_data['is_train'] == False].groupby('cpt_id')[['x_coord', 'y_coord']].first()
            if len(test_locs) > 0:
                ax1.scatter(
                    test_locs['x_coord'],
                    test_locs['y_coord'],
                    [Z_pred.min() for _ in range(len(test_locs))],
                    color='red',
                    marker='o',
                    s=100,
                    label='Testing CPTs'
                )
                
                ax2.scatter(
                    test_locs['x_coord'],
                    test_locs['y_coord'],
                    [Z_real.min() for _ in range(len(test_locs))],
                    color='red',
                    marker='o',
                    s=100,
                    label='Testing CPTs'
                )
            
            ax1.legend()
            ax2.legend()
        else:
            # Original behavior without train/test distinction
            cpt_locations = cpt_data.groupby('cpt_id')[['x_coord', 'y_coord']].first()
            ax1.scatter(
                cpt_locations['x_coord'],
                cpt_locations['y_coord'],
                [Z_pred.min() for _ in range(len(cpt_locations))],
                color='black',
                marker='^',
                s=100,
                label='CPT Locations'
            )
            
            ax2.scatter(
                cpt_locations['x_coord'],
                cpt_locations['y_coord'],
                [Z_real.min() for _ in range(len(cpt_locations))],
                color='black',
                marker='^',
                s=100,
                label='CPT Locations'
            )
            
            ax1.legend()
            ax2.legend()
        
        # Set labels and title
        ax1.set_xlabel('X Coordinate (m)')
        ax1.set_ylabel('Y Coordinate (m)')
        ax1.set_zlabel('Depth (m)')
        ax1.set_title('Predicted Soil Model')
        
        ax2.set_xlabel('X Coordinate (m)')
        ax2.set_ylabel('Y Coordinate (m)')
        ax2.set_zlabel('Depth (m)')
        ax2.set_title('Real Soil Model')
        
        # Invert z-axis for depth
        ax1.invert_zaxis()
        ax2.invert_zaxis()
        
        # Add color bars
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, ticks=unique_soil_types)
        cbar1.set_label('Predicted Soil Type')
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, ticks=unique_soil_types)
        cbar2.set_label('Real Soil Type')
        
        # Set tick labels with abbreviations
        tick_labels = [f"{st} - {SoilTypeManager.get_abbreviation(st)}" for st in unique_soil_types]
        cbar1.ax.set_yticklabels(tick_labels)
        cbar2.ax.set_yticklabels(tick_labels)
        
        plt.suptitle(f'Comparison of Predicted vs Real 3D Soil Models ({dataset_type})', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        return fig