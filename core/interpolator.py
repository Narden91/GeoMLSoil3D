import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import cKDTree


def create_3d_interpolation(data, resolution=10, is_train_only=True, soil_col='predicted_soil'):
    """
    Create a 3D interpolation of soil types from discrete CPT points
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the CPT data with soil types
    resolution : int
        Resolution of the interpolation grid (smaller = higher resolution)
    is_train_only : bool
        Flag indicating if only training data is being used
    soil_col : str
        Column name containing soil types to interpolate
        
    Returns:
    --------
    dict
        Dictionary containing interpolation results including grid data and interpolator
    """
    dataset_type = "training data only" if is_train_only else "all data (training + testing)"
    print(f"Creating 3D soil interpolation using {dataset_type} with '{soil_col}' column...")
    
    # Verifica che la colonna esista
    if soil_col not in data.columns:
        raise ValueError(f"Required column '{soil_col}' not found in data.")
    
    # Extract coordinates and soil types
    points = data[['x_coord', 'y_coord', data.columns[0]]].values  # Using first column as depth
    values = data[soil_col].values
    
    # Check if all x coordinates are the same
    x_coords = data['x_coord'].unique()
    y_coords = data['y_coord'].unique()
    
    print(f"Unique x coordinates: {len(x_coords)}")
    print(f"Unique y coordinates: {len(y_coords)}")
    
    # If we have a degenerate case (all points have same x or y), we need to add some jitter
    if len(x_coords) < 2 or len(y_coords) < 2:
        print("WARNING: Not enough unique coordinates for interpolation. Adding artificial jitter...")
        
        # Create a copy to avoid modifying original data
        jittered_points = points.copy()
        
        # Add small random values to make coordinates unique
        # Scale jitter based on the depth range to maintain reasonable proportions
        depth_range = data[data.columns[0]].max() - data[data.columns[0]].min()
        jitter_scale = depth_range * 0.01  # 1% of depth range
        
        if len(x_coords) < 2:
            jittered_points[:, 0] += np.random.uniform(-jitter_scale, jitter_scale, size=jittered_points.shape[0])
            print(f"Added jitter to x coordinates with scale {jitter_scale}")
            
        if len(y_coords) < 2:
            jittered_points[:, 1] += np.random.uniform(-jitter_scale, jitter_scale, size=jittered_points.shape[0])
            print(f"Added jitter to y coordinates with scale {jitter_scale}")
        
        # Replace original points with jittered points
        points = jittered_points
    
    # Try to create the interpolator - first with LinearNDInterpolator, but fallback to griddata if it fails
    try:
        print("Attempting to create LinearNDInterpolator...")
        interpolator = LinearNDInterpolator(points, values, fill_value=-1)
        interpolator_type = "LinearNDInterpolator"
    except Exception as e:
        print(f"LinearNDInterpolator failed: {e}")
        print("Falling back to griddata interpolator...")
        
        # griddata is more robust but slower
        def griddata_interpolator(pts):
            return griddata(points, values, pts, method='linear', fill_value=-1)
        
        interpolator = griddata_interpolator
        interpolator_type = "griddata"
    
    print(f"Using {interpolator_type} for interpolation")
    
    # Create a grid for interpolation
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    # Add some margin
    margin = 5  # meters
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    # Ensure we have a reasonable range even with jittered values
    if x_max - x_min < resolution:
        center = (x_max + x_min) / 2
        x_min = center - resolution
        x_max = center + resolution
    
    if y_max - y_min < resolution:
        center = (y_max + y_min) / 2
        y_min = center - resolution
        y_max = center + resolution
    
    # Create grid
    x_range = np.arange(x_min, x_max, resolution)
    y_range = np.arange(y_min, y_max, resolution)
    z_range = np.arange(z_min, z_max, resolution/2)  # Higher vertical resolution
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    
    # Flatten the grid
    grid_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    # Interpolate soil types
    print(f"Interpolating {len(grid_points)} points...")
    
    # Handle different interpolator types
    if interpolator_type == "LinearNDInterpolator":
        grid_values = interpolator(grid_points)
    else:
        grid_values = interpolator(grid_points)
    
    # Reshape back to grid
    grid_values = grid_values.reshape(X.shape)
    
    return {
        'grid_data': {
            'X': X,
            'Y': Y, 
            'Z': Z,
            'values': grid_values
        },
        'interpolator': interpolator
    }


def create_3d_interpolation_alternative(data, resolution=10, is_train_only=True, soil_col='predicted_soil'):
    """
    Create a 3D interpolation of soil types using nearest neighbor method
    which is more robust than LinearNDInterpolator for problematic datasets
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing the CPT data with predicted soil types
    resolution : int
        Resolution of the interpolation grid (smaller = higher resolution)
    is_train_only : bool
        Flag indicating if only training data is being used
        
    Returns:
    --------
    dict
        Dictionary containing interpolation results including grid data and interpolator
    """
    dataset_type = "training data only" if is_train_only else "all data (training + testing)"
    print(f"Creating 3D soil interpolation using {dataset_type} with '{soil_col}' column...")
    
    # Get the soil column to use - MODIFICATO per usare 'predicted_soil'
    if soil_col not in data.columns:
        raise ValueError(f"Required column '{soil_col}' not found in data.")
    
    # Verifica che la colonna esista
    if soil_col not in data.columns:
        raise ValueError(f"Required column '{soil_col}' not found in data. Make sure predict_soil_types() was called.")
    
    # Extract coordinates and soil types
    depth_col = data.columns[0]
    coords_df = data[['x_coord', 'y_coord', depth_col]].copy()
    values = data[soil_col].values
    
    # Check coordinate diversity
    x_unique = len(coords_df['x_coord'].unique())
    y_unique = len(coords_df['y_coord'].unique())
    print(f"Dataset has {x_unique} unique x-coordinates and {y_unique} unique y-coordinates")
    
    # Create grid for interpolation
    x_min, x_max = coords_df['x_coord'].min(), coords_df['x_coord'].max()
    y_min, y_max = coords_df['y_coord'].min(), coords_df['y_coord'].max()
    z_min, z_max = coords_df[depth_col].min(), coords_df[depth_col].max()
    
    # Ensure we have some range in the coordinates
    if x_max == x_min:
        print("All x-coordinates are identical. Adding artificial range...")
        range_value = max(10, z_max - z_min)  # Use depth range as a reference
        x_min -= range_value / 2
        x_max += range_value / 2
    
    if y_max == y_min:
        print("All y-coordinates are identical. Adding artificial range...")
        range_value = max(10, z_max - z_min)  # Use depth range as a reference
        y_min -= range_value / 2
        y_max += range_value / 2
    
    # Add some margin
    margin = 5  # meters
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    # Create grid with specified resolution
    x_range = np.linspace(x_min, x_max, int((x_max - x_min) / resolution) + 1)
    y_range = np.linspace(y_min, y_max, int((y_max - y_min) / resolution) + 1)
    z_range = np.linspace(z_min, z_max, int((z_max - z_min) / (resolution/2)) + 1)  # Higher vertical resolution
    
    print(f"Created grid with dimensions: {len(x_range)}x{len(y_range)}x{len(z_range)}")
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    
    # Prepare for nearest neighbor interpolation
    # Create KD-tree from the source points
    source_points = coords_df[['x_coord', 'y_coord', depth_col]].values
    tree = cKDTree(source_points)
    
    # Prepare target points (flatten the grid)
    target_points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()]).T
    
    print(f"Finding nearest neighbors for {len(target_points)} grid points...")
    
    # Find nearest neighbors
    distances, indices = tree.query(target_points, k=1)
    
    # Get soil values for grid points
    grid_values = values[indices]
    
    # Reshape back to grid
    grid_values = grid_values.reshape(X.shape)
    
    # Create a "fake" interpolator function for compatibility with existing code
    def nn_interpolator(points):
        """Nearest neighbor interpolator function"""
        _, idx = tree.query(points, k=1)
        return values[idx]
    
    return {
        'grid_data': {
            'X': X,
            'Y': Y, 
            'Z': Z,
            'values': grid_values
        },
        'interpolator': nn_interpolator
    }