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
            # Load file and preprocess
            df = _load_and_preprocess_file(file_path, i, x_coord_col, y_coord_col, is_train)
            all_data.append(df)
            print(f"Loaded {len(df)} records from {os.path.basename(file_path)} for {dataset_type}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Combine all data
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"Combined {dataset_type} dataset contains {len(combined_data)} records")
        
        # Check for coordinate issues
        _check_coordinates(combined_data, dataset_type)
        
        return combined_data
    else:
        raise ValueError(f"No data could be loaded for {dataset_type} set")


def _load_and_preprocess_file(file_path, index, x_coord_col=None, y_coord_col=None, is_train=True):
    """
    Load and preprocess a single CPT file
    
    Parameters:
    -----------
    file_path : str
        Path to the file
    index : int
        Index of the file in the sequence
    x_coord_col : str, optional
        Column name containing X coordinates
    y_coord_col : str, optional
        Column name containing Y coordinates
    is_train : bool
        Flag to indicate if this is training data
        
    Returns:
    --------
    DataFrame
        Preprocessed data from the file
    """
    # Extract file name without extension
    file_name = os.path.basename(file_path).split('.')[0]
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Add source file identifier
    df['cpt_id'] = file_name
    
    # Add dataset type flag
    df['is_train'] = is_train
    
    # Handle coordinates
    df = _handle_coordinates(df, index, x_coord_col, y_coord_col, file_name)
    
    # Handle soil type classification
    df = _handle_soil_classification(df, file_name)
    
    return df


def _handle_coordinates(df, index, x_coord_col=None, y_coord_col=None, file_name=""):
    """
    Handle coordinate columns in the dataframe.
    Assigns a unique (x, y) coordinate pair to all points within the same CPT file
    if coordinates are not found in the source file.
    """
    # If X and Y coordinates are not in the data, we need to add them
    if (x_coord_col is None or y_coord_col is None or
        x_coord_col not in df.columns or y_coord_col not in df.columns):

        print(f"No coordinates found for {file_name}. Generating unique spatial coordinates for this CPT.")

        # Generate a single, unique (x, y) coordinate pair for the entire CPT file
        # using a spiral pattern based on the file index.
        # We assume a maximum of 100 files for the spiral generation density,
        # adjust if necessary based on expected dataset size.
        num_files_for_spiral = 100 # Or pass the actual total number of files if available
        x, y = generate_spiral_coords(index, num_files_for_spiral)

        # Assign the *same* generated (x, y) coordinate to all rows in this DataFrame (CPT file).
        # Do NOT add per-point variations/jitter, as a CPT sounding occurs at a single (x, y) location.
        df['x_coord'] = x
        df['y_coord'] = y

        print(f"Assigned generated coordinates ({x:.2f}, {y:.2f}) to all points in {file_name}")

    else:
        # Coordinates exist in the file, ensure they use the standard names.
        print(f"Using coordinates from columns '{x_coord_col}' and '{y_coord_col}' for {file_name}.")
        df.rename(columns={x_coord_col: 'x_coord', y_coord_col: 'y_coord'}, inplace=True)

        # Optional: Check if coordinates are constant within the file, as expected for a CPT
        if df['x_coord'].nunique() > 1 or df['y_coord'].nunique() > 1:
            print(f"WARNING: Multiple (x, y) coordinates found within the single CPT file {file_name}. "
                  f"Using the provided coordinates, but this might indicate data issues.")

    return df


def _handle_soil_classification(df, file_name=""):
    """
    Handle soil classification in the dataframe
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    file_name : str
        Name of the file for logging
        
    Returns:
    --------
    DataFrame
        Dataframe with soil classification
    """
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
    
    return df


def _check_coordinates(df, dataset_type=""):
    """
    Check for coordinate issues in the dataframe
    
    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    dataset_type : str
        Type of dataset for logging
    """
    if len(df['x_coord'].unique()) < 2:
        print(f"WARNING: All x coordinates are identical in {dataset_type} set. 3D interpolation may fail.")
    
    if len(df['y_coord'].unique()) < 2:
        print(f"WARNING: All y coordinates are identical in {dataset_type} set. 3D interpolation may fail.")