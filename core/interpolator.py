import numpy as np
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import cKDTree


def create_3d_interpolation(data, resolution=10, is_train_only=True, soil_col='predicted_soil', method='linear', fixed_bounds=None):
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
    method : str
        Interpolation method ('linear' or 'nearest')
    fixed_bounds : tuple, optional
        Fixed bounds (x_min, x_max, y_min, y_max, z_min, z_max) to use for grid creation
        
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
    depth_col = data.columns[0]
    coords_df = data[['x_coord', 'y_coord', depth_col]].copy()
    values = data[soil_col].values
    
    # Stampa i valori unici di x_coord e y_coord per debug
    print(f"Unique x_coord values: {sorted(coords_df['x_coord'].unique())}")
    print(f"Unique y_coord values: {sorted(coords_df['y_coord'].unique())}")
    
    # Check coordinate diversity
    x_unique = len(coords_df['x_coord'].unique())
    y_unique = len(coords_df['y_coord'].unique())
    print(f"Dataset has {x_unique} unique x-coordinates and {y_unique} unique y-coordinates")
    
    # Get bounds with sanity checks - use fixed bounds if provided
    if fixed_bounds:
        x_min, x_max, y_min, y_max, z_min, z_max = fixed_bounds
        print(f"Using fixed bounds: X({x_min}, {x_max}), Y({y_min}, {y_max}), Z({z_min}, {z_max})")
    else:
        x_min, x_max, y_min, y_max, z_min, z_max = _get_interpolation_bounds(coords_df, depth_col)
    
    # Create grid for interpolation
    x_range, y_range, z_range = _create_grid_ranges(x_min, x_max, y_min, y_max, z_min, z_max, resolution)
    print(f"Created grid with dimensions: {len(x_range)}x{len(y_range)}x{len(z_range)}")
    
    # Create meshgrid
    X, Y, Z = np.meshgrid(x_range, y_range, z_range)
    
    # Create interpolator and grid values based on method
    if method.lower() == 'nearest':
        grid_values, interpolator = _create_nearest_interpolation(
            coords_df[['x_coord', 'y_coord', depth_col]].values, 
            values, 
            X, Y, Z
        )
    else:  # Default to 'linear'
        grid_values, interpolator = _create_linear_interpolation(
            coords_df[['x_coord', 'y_coord', depth_col]].values, 
            values, 
            X, Y, Z
        )
    
    # Store the bounds used for this interpolation for reference
    bounds_used = (x_min, x_max, y_min, y_max, z_min, z_max)
    
    # Per garantire compatibilità con entrambi gli stili di accesso ai dati,
    # restituiamo sia la struttura 'grid_data' nidificata che gli attributi diretti
    result = {
        'grid_data': {
            'X': X,
            'Y': Y, 
            'Z': Z,
            'values': grid_values
        },
        'interpolator': interpolator,
        'bounds': bounds_used,
        # Aggiungiamo anche l'accesso diretto per retrocompatibilità
        'X': X,
        'Y': Y,
        'Z': Z,
        'values': grid_values
    }
    
    return result


def _get_interpolation_bounds(coords_df, depth_col, margin=5):
    """
    Calculate bounds for interpolation with margin and sanity checks
    
    Parameters:
    -----------
    coords_df : pandas.DataFrame
        DataFrame with coordinate columns
    depth_col : str
        Name of depth column
    margin : float
        Margin to add around the bounds
        
    Returns:
    --------
    x_min, x_max, y_min, y_max, z_min, z_max : tuple
        Bounds for interpolation
    """
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
    
    # Add margin
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin
    
    return x_min, x_max, y_min, y_max, z_min, z_max


def _create_grid_ranges(x_min, x_max, y_min, y_max, z_min, z_max, resolution):
    """
    Create grid ranges for interpolation
    
    Parameters:
    -----------
    x_min, x_max, y_min, y_max, z_min, z_max : float
        Bounds for interpolation
    resolution : int
        Resolution of the grid
        
    Returns:
    --------
    x_range, y_range, z_range : tuple
        Grid ranges for interpolation
    """
    x_range = np.linspace(x_min, x_max, int((x_max - x_min) / resolution) + 1)
    y_range = np.linspace(y_min, y_max, int((y_max - y_min) / resolution) + 1)
    z_range = np.linspace(z_min, z_max, int((z_max - z_min) / (resolution/2)) + 1)  # Higher vertical resolution
    
    return x_range, y_range, z_range


def _create_linear_interpolation(points, values, X, Y, Z):
    """
    Create linear interpolation
    
    Parameters:
    -----------
    points : ndarray
        Points to interpolate from (x, y, z)
    values : ndarray
        Values at the points
    X, Y, Z : ndarray
        Meshgrid for interpolation
        
    Returns:
    --------
    grid_values, interpolator : tuple
        Interpolated values and interpolator function
    """
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
    
    # Flatten the grid for interpolation
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
    
    return grid_values, interpolator


def _create_nearest_interpolation(points, values, X, Y, Z):
    """
    Create nearest neighbor interpolation
    
    Parameters:
    -----------
    points : ndarray
        Points to interpolate from (x, y, z)
    values : ndarray
        Values at the points
    X, Y, Z : ndarray
        Meshgrid for interpolation
        
    Returns:
    --------
    grid_values, interpolator : tuple
        Interpolated values and interpolator function
    """
    # Prepare for nearest neighbor interpolation
    # Create KD-tree from the source points
    tree = cKDTree(points)
    
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
    
    return grid_values, nn_interpolator


# Funzione wrapper per compatibilità con il codice esistente
def create_3d_interpolation_alternative(data, resolution=10, is_train_only=True, soil_col='predicted_soil', fixed_bounds=None):
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
    soil_col : str
        Column name containing soil types to interpolate
    fixed_bounds : tuple, optional
        Fixed bounds (x_min, x_max, y_min, y_max, z_min, z_max) to use for grid creation
        
    Returns:
    --------
    dict
        Dictionary containing interpolation results including grid data and interpolator
    """
    return create_3d_interpolation(data, resolution, is_train_only, soil_col, method='nearest', fixed_bounds=fixed_bounds)