import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.stats import chisquare
from utils.soil_types import SoilTypeManager


def plot_test_vs_predicted(y_test, y_pred, soil_types=None):
    """
    Plot confusion matrix and other evaluation metrics comparing test vs predicted values
    
    Parameters:
    -----------
    y_test : array-like
        True soil types from test set
    y_pred : array-like
        Predicted soil types
    soil_types : list, optional
        List of all possible soil types (if None, will be inferred from data)
    """
    if soil_types is None:
        soil_types = sorted(set(np.concatenate((y_test, y_pred))))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=soil_types)
    
    # Create tick labels with abbreviations
    tick_labels = [f"{st} ({SoilTypeManager.get_abbreviation(st)})" for st in soil_types]
    
    # Plot normalized confusion matrix
    _plot_normalized_confusion_matrix(cm, soil_types, tick_labels)
    
    # Plot raw counts heatmap
    _plot_raw_confusion_matrix(cm, soil_types, tick_labels)
    
    # Plot accuracy by soil type
    _plot_accuracy_by_soil_type(y_test, y_pred, soil_types, tick_labels)


def _plot_normalized_confusion_matrix(cm, soil_types, tick_labels):
    """
    Plot normalized confusion matrix
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    soil_types : list
        List of soil types
    tick_labels : list
        List of tick labels
    """
    # Normalize by row (true values) with gestione del caso di riga con somma zero
    cm_norm = np.zeros_like(cm, dtype=float)
    
    # Per ogni riga, normalizza solo se la somma è maggiore di zero
    row_sums = cm.sum(axis=1)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_norm[i, :] = cm[i, :] / row_sums[i]
        # Le righe con somma zero rimarranno a zero
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.ylabel('True Soil Type')
    plt.xlabel('Predicted Soil Type')
    plt.title('Normalized Confusion Matrix for Test Set')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()


def _plot_raw_confusion_matrix(cm, soil_types, tick_labels):
    """
    Plot raw confusion matrix
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    soil_types : list
        List of soil types
    tick_labels : list
        List of tick labels
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=tick_labels, yticklabels=tick_labels)
    plt.ylabel('True Soil Type')
    plt.xlabel('Predicted Soil Type')
    plt.title('Confusion Matrix for Test Set (Counts)')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()


def _plot_accuracy_by_soil_type(y_test, y_pred, soil_types, tick_labels):
    """
    Plot accuracy by soil type
    
    Parameters:
    -----------
    y_test : array-like
        True soil types
    y_pred : array-like
        Predicted soil types
    soil_types : list
        List of soil types
    tick_labels : list
        List of tick labels
    """
    # Calculate accuracy and counts by soil type
    accuracies, counts = _calculate_accuracy_by_soil_type(y_test, y_pred, soil_types)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot accuracies
    bars = ax1.bar(range(len(soil_types)), accuracies, align='center')
    
    # Add data labels
    for bar, acc in zip(bars, accuracies):
        if acc > 0:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{acc:.2f}', ha='center', va='bottom')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Set Accuracy by Soil Type')
    ax1.set_ylim(0, 1.1)
    
    # Plot sample counts
    ax2.bar(range(len(soil_types)), counts, align='center', color='lightgreen')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Test Set Sample Count by Soil Type')
    
    # Set common x-axis labels
    ax2.set_xticks(range(len(soil_types)))
    ax2.set_xticklabels(tick_labels, rotation=45)
    ax2.set_xlabel('Soil Type')
    
    plt.tight_layout()
    plt.show()


def _calculate_accuracy_by_soil_type(y_test, y_pred, soil_types):
    """
    Calculate accuracy by soil type
    
    Parameters:
    -----------
    y_test : array-like
        True soil types
    y_pred : array-like
        Predicted soil types
    soil_types : list
        List of soil types
        
    Returns:
    --------
    accuracies, counts : tuple
        Lists of accuracies and sample counts by soil type
    """
    accuracies = []
    counts = []
    
    for soil_type in soil_types:
        mask = (y_test == soil_type)
        count = mask.sum()
        if count > 0:
            accuracy = (y_pred[mask] == soil_type).mean()
            accuracies.append(accuracy)
            counts.append(count)
        else:
            accuracies.append(0)
            counts.append(0)
    
    return accuracies, counts


def plot_performance_by_cpt(test_data):
    """
    Plot performance metrics for each CPT file in the test set
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with actual and predicted soil types
    """
    if 'predicted_soil' not in test_data.columns or 'soil []' not in test_data.columns:
        raise ValueError("Test data must contain 'predicted_soil' and 'soil []' columns")
    
    # Calculate accuracy by CPT file
    cpt_accuracies, cpt_ids, sample_counts = _calculate_performance_by_cpt(test_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot accuracies
    bars = ax1.bar(range(len(cpt_ids)), cpt_accuracies, align='center')
    
    # Add data labels
    for bar, acc in zip(bars, cpt_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.2f}', ha='center', va='bottom')
    
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Set Accuracy by CPT File')
    ax1.set_ylim(0, 1.1)
    
    # Plot sample counts
    ax2.bar(range(len(cpt_ids)), sample_counts, align='center', color='lightgreen')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Test Set Sample Count by CPT File')
    
    # Set common x-axis labels
    ax2.set_xticks(range(len(cpt_ids)))
    ax2.set_xticklabels(cpt_ids, rotation=45)
    ax2.set_xlabel('CPT ID')
    
    plt.tight_layout()
    plt.show()


def _calculate_performance_by_cpt(test_data):
    """
    Calculate performance metrics by CPT file
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with actual and predicted soil types
        
    Returns:
    --------
    cpt_accuracies, cpt_ids, sample_counts : tuple
        Lists of accuracies, CPT IDs, and sample counts
    """
    cpt_accuracies = []
    cpt_ids = []
    sample_counts = []
    
    for cpt_id in test_data['cpt_id'].unique():
        mask = (test_data['cpt_id'] == cpt_id)
        cpt_data = test_data[mask]
        accuracy = (cpt_data['predicted_soil'] == cpt_data['soil []']).mean()
        
        cpt_accuracies.append(accuracy)
        cpt_ids.append(cpt_id)
        sample_counts.append(len(cpt_data))
    
    # Sort by accuracy
    sorted_indices = np.argsort(cpt_accuracies)
    cpt_accuracies = [cpt_accuracies[i] for i in sorted_indices]
    cpt_ids = [cpt_ids[i] for i in sorted_indices]
    sample_counts = [sample_counts[i] for i in sorted_indices]
    
    return cpt_accuracies, cpt_ids, sample_counts


def plot_depth_vs_accuracy(test_data, depth_bins=10):
    """
    Plot accuracy as a function of depth
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with actual and predicted soil types
    depth_bins : int
        Number of depth bins to create
    """
    if 'predicted_soil' not in test_data.columns or 'soil []' not in test_data.columns:
        raise ValueError("Test data must contain 'predicted_soil' and 'soil []' columns")
    
    # Get depth column (assumed to be first column)
    depth_col = test_data.columns[0]
    
    # Create depth bins and calculate accuracy
    bin_centers, accuracies, sample_counts = _calculate_accuracy_by_depth(
        test_data, depth_col, depth_bins
    )
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot accuracies
    ax1.plot(bin_centers, accuracies, 'o-', linewidth=2)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Test Set Accuracy by Depth')
    ax1.set_ylim(0, 1.1)
    ax1.grid(True)
    
    # Get bin edges for histogram
    depth_min = test_data[depth_col].min()
    depth_max = test_data[depth_col].max()
    bin_edges = np.linspace(depth_min, depth_max, depth_bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Plot sample counts
    ax2.bar(bin_centers, sample_counts, align='center', width=bin_width*0.8, color='lightgreen')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Test Set Sample Count by Depth')
    
    # Set common x-axis labels
    ax2.set_xlabel('Depth (m)')
    ax2.invert_xaxis()  # Typically depth increases to the right
    
    plt.tight_layout()
    plt.show()


def _calculate_accuracy_by_depth(test_data, depth_col, depth_bins):
    """
    Calculate accuracy by depth bin
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with actual and predicted soil types
    depth_col : str
        Name of depth column
    depth_bins : int
        Number of depth bins
        
    Returns:
    --------
    bin_centers, accuracies, sample_counts : tuple
        Lists of bin centers, accuracies, and sample counts
    """
    # Create depth bins
    depth_min = test_data[depth_col].min()
    depth_max = test_data[depth_col].max()
    bin_edges = np.linspace(depth_min, depth_max, depth_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate accuracy by depth bin
    accuracies = []
    sample_counts = []
    
    for i in range(depth_bins):
        bin_mask = (test_data[depth_col] >= bin_edges[i]) & (test_data[depth_col] < bin_edges[i+1])
        bin_data = test_data[bin_mask]
        
        if len(bin_data) > 0:
            accuracy = (bin_data['predicted_soil'] == bin_data['soil []']).mean()
            accuracies.append(accuracy)
            sample_counts.append(len(bin_data))
        else:
            accuracies.append(0)
            sample_counts.append(0)
    
    return bin_centers, accuracies, sample_counts


def plot_soil_distribution_comparison(train_data, test_data):
    """
    Plot comparison of soil type distributions between training and test sets
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with soil types
    test_data : pandas.DataFrame
        Test data with soil types
    """
    if 'soil []' not in train_data.columns or 'soil []' not in test_data.columns:
        raise ValueError("Data must contain 'soil []' column")
    
    # Count soil types in both datasets
    train_counts, test_counts, all_types = _prepare_soil_distribution_data(train_data, test_data)
    
    # Create tick labels with abbreviations
    tick_labels = [f"{st} ({SoilTypeManager.get_abbreviation(st)})" for st in all_types]
    
    # Plot distribution comparison
    _plot_soil_distribution_comparison(train_counts, test_counts, all_types, tick_labels)
    
    # Calculate and print chi-square statistic
    _calculate_distribution_similarity(train_counts, test_counts, all_types)


def _prepare_soil_distribution_data(train_data, test_data):
    """
    Prepare soil distribution data for comparison
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with soil types
    test_data : pandas.DataFrame
        Test data with soil types
        
    Returns:
    --------
    train_counts, test_counts, all_types : tuple
        Counts for training and test sets, and list of all soil types
    """
    # Count soil types in both datasets
    train_counts = train_data['soil []'].value_counts().sort_index()
    test_counts = test_data['soil []'].value_counts().sort_index()
    
    # Ensure all soil types are represented in both counts
    all_types = sorted(set(train_counts.index) | set(test_counts.index))
    
    train_counts_full = pd.Series([train_counts.get(t, 0) for t in all_types], index=all_types)
    test_counts_full = pd.Series([test_counts.get(t, 0) for t in all_types], index=all_types)
    
    return train_counts_full, test_counts_full, all_types


def _plot_soil_distribution_comparison(train_counts, test_counts, all_types, tick_labels):
    """
    Plot soil distribution comparison
    
    Parameters:
    -----------
    train_counts : Series
        Soil type counts for training set
    test_counts : Series
        Soil type counts for test set
    all_types : list
        List of all soil types
    tick_labels : list
        List of tick labels
    """
    # Convert to percentages
    train_pct = train_counts / train_counts.sum() * 100
    test_pct = test_counts / test_counts.sum() * 100
    
    # Create grouped bar chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot count comparison
    x = np.arange(len(all_types))
    width = 0.35
    
    ax1.bar(x - width/2, train_counts, width, label='Training Set')
    ax1.bar(x + width/2, test_counts, width, label='Test Set')
    
    ax1.set_ylabel('Count')
    ax1.set_title('Soil Type Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax1.legend()
    
    # Plot percentage comparison
    ax2.bar(x - width/2, train_pct, width, label='Training Set')
    ax2.bar(x + width/2, test_pct, width, label='Test Set')
    
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Soil Type Distribution Comparison (Percentage)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def _calculate_distribution_similarity(train_counts, test_counts, all_types):
    """
    Calculate similarity between distributions using chi-square test
    
    Parameters:
    -----------
    train_counts : Series
        Soil type counts for training set
    test_counts : Series
        Soil type counts for test set
    all_types : list
        List of all soil types
    """
    # Scale test counts to same total as train counts for comparison
    scale_factor = train_counts.sum() / test_counts.sum()
    test_counts_scaled = test_counts * scale_factor
    
    # Calculate chi-square statistic
    chi2_stat, p_value = chisquare(test_counts_scaled, train_counts)
    
    print(f"Chi-square test for distribution similarity: stat={chi2_stat:.4f}, p-value={p_value:.4f}")
    print(f"Interpretation: {'Similar distributions' if p_value > 0.05 else 'Different distributions'} (at alpha=0.05)")