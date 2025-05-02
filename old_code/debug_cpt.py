import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

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
        file_name = os.path.basename(file_path)
        print(f"\n{'='*50}\nAnalyzing file {i+1}/{len(file_paths)}: {file_name}\n{'='*50}")
        
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Basic info
            print(f"Shape: {df.shape}")
            print(f"Columns: {', '.join(df.columns)}")
            
            # Check for coordinate columns
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
            
            # Check the first column (usually depth)
            depth_col = df.columns[0]
            print(f"\nDepth column (assumed): {depth_col}")
            print(f"Depth range: {df[depth_col].min()} to {df[depth_col].max()}")
            
            # Check for soil classification column
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
            
            # Check for common CPT parameters
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
            
            # Plot depth vs. main CPT parameters for visual inspection
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
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    print("\nDebug analysis complete")
