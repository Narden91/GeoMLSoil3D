import pandas as pd
import numpy as np
from glob import glob
import os
import random

from utils.helpers import generate_spiral_coords
from utils.soil_types import SoilTypeManager


def split_cpt_files(file_pattern, test_size=0.2, random_state=42):
    """
    Split CPT files into training and testing sets
    
    Parameters:
    -----------
    file_pattern : str
        File pattern to match CPT files (e.g., "data/CPT_*.csv")
    test_size : float
        Proportion of files to use for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    train_files, test_files : tuple of lists
        Lists containing file paths for training and testing
    """
    # Find all matching files
    file_paths = glob(file_pattern)
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Found {len(file_paths)} CPT files")
    
    # Set random seed for reproducibility
    random.seed(random_state)
    
    # Shuffle files
    shuffled_files = file_paths.copy()
    random.shuffle(shuffled_files)
    
    # Calculate number of test files
    n_test = max(1, int(len(shuffled_files) * test_size))
    
    # Split files
    test_files = shuffled_files[:n_test]
    train_files = shuffled_files[n_test:]
    
    return train_files, test_files


def load_cpt_files(file_paths, x_coord_col=None, y_coord_col=None, is_train=True):
    """
    Load CPT data from specified files
    
    Parameters:
    -----------
    file_paths : list
        List of file paths to load
    x_coord_col : str, optional
        Column name containing X coordinates
    y_coord_col : str, optional
        Column name containing Y coordinates
    is_train : bool
        Flag to indicate if this is training data (for logging)
        
    Returns:
    --------
    DataFrame
        Combined data from the specified files
    """
    dataset_type = "training" if is_train else "testing"
    all_data = []
    
    for i, file_path in enumerate(file_paths):
        try:
            # Extract file name without extension
            file_name = os.path.basename(file_path).split('.')[0]
            
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Add source file identifier
            df['cpt_id'] = file_name
            
            # Add dataset type flag
            df['is_train'] = is_train
            
            # If X and Y coordinates are not in the data, we need to add them
            if (x_coord_col is None or y_coord_col is None or 
                x_coord_col not in df.columns or y_coord_col not in df.columns):
                
                print(f"No coordinates for {file_name}, generating spatial coordinates")
                
                # Generate coordinates using spiral pattern
                x, y = generate_spiral_coords(i, len(file_paths))
                
                # Add some small variations for each point within the same CPT
                # This is important to avoid the "same coordinate" problem
                point_count = len(df)
                random_state = np.random.RandomState(i)  # Use file index as seed for reproducibility
                
                # Generate small variations (max 1 meter in each direction)
                x_variations = random_state.uniform(-0.5, 0.5, size=point_count)
                y_variations = random_state.uniform(-0.5, 0.5, size=point_count)
                
                # Create artificial X and Y coordinates with variations
                df['x_coord'] = x + x_variations
                df['y_coord'] = y + y_variations
                
                print(f"Generated coordinates centered at ({x:.2f}, {y:.2f}) with variations")
            else:
                # Rename to standard column names
                df.rename(columns={x_coord_col: 'x_coord', y_coord_col: 'y_coord'}, inplace=True)
            
            # Codificare il tipo di suolo se qc e Rf sono presenti nei dati
            if 'qc [MPa]' in df.columns and 'Rf [%]' in df.columns and 'soil []' not in df.columns:
                print(f"Computing soil types for {file_name} based on qc and Rf")
                df['soil []'] = df.apply(
                    lambda row: SoilTypeManager.code_from_qc_rf(row['qc [MPa]'], row['Rf [%]']), 
                    axis=1
                )
            
            # Aggiungere le abbreviazioni e le descrizioni dei tipi di suolo
            if 'soil []' in df.columns:
                df = SoilTypeManager.convert_dataset_labels(df)
            
            all_data.append(df)
            print(f"Loaded {len(df)} records from {file_name} for {dataset_type}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined {dataset_type} dataset contains {len(combined_data)} records")
        
        # Check if coordinates are all the same and warn
        if len(combined_data['x_coord'].unique()) < 2:
            print(f"WARNING: All x coordinates are identical in {dataset_type} set. 3D interpolation may fail.")
        
        if len(combined_data['y_coord'].unique()) < 2:
            print(f"WARNING: All y coordinates are identical in {dataset_type} set. 3D interpolation may fail.")
        
        return combined_data
    else:
        raise ValueError(f"No data could be loaded for {dataset_type} set")