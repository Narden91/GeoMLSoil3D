import numpy as np
import matplotlib.cm as cm
import matplotlib as mpl
from glob import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


def generate_spiral_coords(index, num_files):
    """
    Generate spatial coordinates in a spiral pattern for CPT points
    
    Parameters:
    -----------
    index : int
        Index of the file in the sequence
    num_files : int
        Total number of files
        
    Returns:
    --------
    x, y : tuple of float
        X and Y coordinates
    """
    # Parameters for spiral
    a = 100  # Controls spacing between rings
    b = 1   # Controls how tightly wound the spiral is
    
    # Convert index to angle and radius
    theta = b * index * 2 * np.pi / num_files
    r = a * theta / (2 * np.pi)

    # Convert to Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    return x, y


def get_compatible_colormap(cmap_name, n_colors):
    """
    Get a colormap compatible with the installed matplotlib version
    
    Parameters:
    -----------
    cmap_name : str
        Name of the colormap
    n_colors : int
        Number of colors to generate
        
    Returns:
    --------
    cmap : matplotlib.colors.Colormap
        Colormap object
    """
    # Try the newer method first (Matplotlib ≥ 3.7)
    try:
        # New method
        cmap = mpl.colormaps[cmap_name].resampled(n_colors)
    except (AttributeError, KeyError):
        # Fallback for older versions
        cmap = cm.get_cmap(cmap_name, n_colors)
    
    return cmap


def debug_cpt_files(file_pattern):
    """
    Debug script to analyze CPT files and their coordinates
    
    Parameters:
    -----------
    file_pattern : str
        File pattern to match CPT files (e.g., "data/CPT_*.csv")
    """
    print(f"Debugging CPT files with pattern: {file_pattern}")
    
    # Find all matching files
    file_paths = glob(file_pattern)
    if not file_paths:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    print(f"Found {len(file_paths)} CPT files")
    
    # Analyze each file
    for i, file_path in enumerate(file_paths):
        analyze_cpt_file(file_path, i, len(file_paths))


def analyze_cpt_file(file_path, index, total_files):
    """
    Analyze a single CPT file
    
    Parameters:
    -----------
    file_path : str
        Path to the CPT file
    index : int
        Index of the file in the sequence
    total_files : int
        Total number of files
    """
    file_name = os.path.basename(file_path)
    print(f"\n{'='*50}\nAnalyzing file {index+1}/{total_files}: {file_name}\n{'='*50}")
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Basic info
        print(f"Shape: {df.shape}")
        print(f"Columns: {', '.join(df.columns)}")
        
        # Check for coordinate columns
        analyze_coordinate_columns(df)
        
        # Check the first column (usually depth)
        analyze_depth_column(df)
        
        # Check for soil classification column
        analyze_soil_columns(df)
        
        # Check for common CPT parameters
        analyze_cpt_parameters(df)
        
        # Plot depth vs. main CPT parameters for visual inspection
        plot_cpt_parameters(df, file_name)
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")


def analyze_coordinate_columns(df):
    """
    Analyze coordinate columns in a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    """
    coord_columns = [col for col in df.columns if 'coord' in col.lower() or 'east' in col.lower() or 
                    'north' in col.lower() or '_x' in col.lower() or '_y' in col.lower()]
    
    if coord_columns:
        print(f"Potential coordinate columns found: {', '.join(coord_columns)}")
        
        for col in coord_columns:
            unique_vals = df[col].nunique()
            print(f"Column '{col}' has {unique_vals} unique values")
            
            # If few unique values, print them
            if unique_vals < 10:
                print(f"Values: {df[col].unique()}")
    else:
        print("No coordinate columns found")


def analyze_depth_column(df):
    """
    Analyze depth column in a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    """
    # Check the first column (usually depth)
    depth_col = df.columns[0]
    print(f"\nDepth column (assumed): {depth_col}")
    print(f"Depth range: {df[depth_col].min()} to {df[depth_col].max()}")


def analyze_soil_columns(df):
    """
    Analyze soil classification columns in a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    """
    soil_cols = [col for col in df.columns if 'soil' in col.lower() or 'class' in col.lower() or 
                'type' in col.lower() or '[]' in col]
    
    if soil_cols:
        print(f"\nPotential soil classification columns: {', '.join(soil_cols)}")
        
        for col in soil_cols:
            unique_vals = df[col].nunique()
            print(f"Column '{col}' has {unique_vals} unique values")
            
            # If few unique values, print them
            if unique_vals < 20:
                value_counts = df[col].value_counts()
                for val, count in value_counts.items():
                    print(f"  Value {val}: {count} occurrences ({count/len(df)*100:.1f}%)")
    else:
        print("\nNo soil classification columns found")


def analyze_cpt_parameters(df):
    """
    Analyze CPT parameter columns in a dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to analyze
    """
    cpt_params = [col for col in df.columns if any(param in col.lower() for param in 
                                                ['qc', 'fs', 'rf', 'friction', 'resistance'])]
    
    if cpt_params:
        print(f"\nCPT parameter columns: {', '.join(cpt_params)}")
        
        # Print statistics for each parameter
        for col in cpt_params:
            stats = df[col].describe()
            print(f"\nStatistics for '{col}':")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
    else:
        print("\nNo standard CPT parameter columns found")


def plot_cpt_parameters(df, file_name):
    """
    Plot CPT parameters vs. depth
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to plot
    file_name : str
        Name of the file for plot title
    """
    # Check for depth column and CPT parameters
    depth_col = df.columns[0]  # Depth column is typically the first column
    cpt_params = [col for col in df.columns if any(param in col.lower() for param in 
                                                ['qc', 'fs', 'rf', 'friction', 'resistance'])]
    
    if len(cpt_params) > 0 and depth_col:
        plt.figure(figsize=(15, 10))
        
        for i, col in enumerate(cpt_params[:3]):  # Plot up to 3 parameters
            plt.subplot(1, min(3, len(cpt_params)), i+1)
            plt.plot(df[col], df[depth_col])
            plt.xlabel(col)
            plt.ylabel(depth_col)
            plt.title(f'{col} vs {depth_col}')
            plt.grid(True)
            plt.gca().invert_yaxis()  # Depth increases downward
        
        plt.tight_layout()
        plt.suptitle(f'CPT Parameters for {file_name}', fontsize=16)
        plt.subplots_adjust(top=0.9)
        plt.show()