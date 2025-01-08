import numpy as np
from itertools import combinations
from typing import List, Dict, Union, Tuple
import pandas as pd

def krippendorffs_alpha(data: Union[np.ndarray, List[List]], metric: str = 'nominal') -> float:
    """
    Calculate Krippendorff's alpha for inter-rater reliability.
    
    Parameters:
    -----------
    data : Union[np.ndarray, List[List]]
        Matrix where rows are units/items and columns are annotators.
        Missing values should be represented as np.nan
    metric : str
        The metric to use for disagreement:
        - 'nominal': for categorical data
        - 'ordinal': for ordered categorical data
        - 'interval': for interval data
        
    Returns:
    --------
    float
        Krippendorff's alpha coefficient (-1 to 1)
        
    Example:
    --------
    >>> data = np.array([
    ...     [1, 1, 1, np.nan],
    ...     [2, 2, 2, 2],
    ...     [3, 3, 3, 3],
    ...     [3, 3, 3, 3],
    ...     [2, 2, 2, 2]
    ... ])
    >>> alpha = krippendorffs_alpha(data)
    >>> print(f"Krippendorff's alpha: {alpha:.3f}")
    """
    
    if isinstance(data, list):
        data = np.array(data)
    
    # Convert to float for handling np.nan
    data = data.astype(float)
    
    def nominal_distance(v1, v2):
        return 1.0 if v1 != v2 else 0.0
    
    def ordinal_distance(v1, v2):
        return (v1 - v2) ** 2
    
    def interval_distance(v1, v2):
        return (v1 - v2) ** 2
    
    metric_functions = {
        'nominal': nominal_distance,
        'ordinal': ordinal_distance,
        'interval': interval_distance
    }
    
    if metric not in metric_functions:
        raise ValueError(f"Metric must be one of {list(metric_functions.keys())}")
    
    distance_function = metric_functions[metric]
    
    # Calculate observed disagreement
    n = 0  # number of pairable elements
    Do = 0  # observed disagreement
    
    for unit in data:
        # Get all valid pairs in this unit
        unit_values = unit[~np.isnan(unit)]
        if len(unit_values) < 2:
            continue
            
        pairs = list(combinations(unit_values, 2))
        n += len(pairs)
        
        for v1, v2 in pairs:
            Do += distance_function(v1, v2)
    
    Do = Do / n if n > 0 else 0
    
    # Calculate expected disagreement
    # Get value counts across all valid ratings
    values = data[~np.isnan(data)]
    value_counts = dict(zip(*np.unique(values, return_counts=True)))
    
    De = 0  # expected disagreement
    total_pairable = sum(value_counts.values())
    
    for v1 in value_counts:
        for v2 in value_counts:
            De += distance_function(v1, v2) * value_counts[v1] * value_counts[v2]
    
    De = De / (total_pairable * (total_pairable - 1))
    
    # Calculate alpha
    alpha = 1 - (Do / De) if De != 0 else 1
    
    return alpha

def generate_pairwise_agreement_report(data: Union[np.ndarray, List[List]], 
                                     annotator_names: List[str] = None,
                                     metric: str = 'nominal') -> pd.DataFrame:
    """
    Generate a report of pairwise agreement between all annotators.
    
    Parameters:
    -----------
    data : Union[np.ndarray, List[List]]
        Matrix where rows are units/items and columns are annotators
    annotator_names : List[str], optional
        List of annotator names/IDs. If None, will use numbers
    metric : str
        The metric to use for calculating agreement
        
    Returns:
    --------
    pd.DataFrame
        Matrix of pairwise agreement scores
        
    Example:
    --------
    >>> data = np.array([
    ...     [1, 1, 1, 2],
    ...     [2, 2, 2, 2],
    ...     [3, 3, 3, 3],
    ...     [3, 3, 2, 3],
    ...     [2, 2, 2, 2]
    ... ])
    >>> annotators = ['Ann1', 'Ann2', 'Ann3', 'Ann4']
    >>> report = generate_pairwise_agreement_report(data, annotators)
    >>> print(report)
    """
    if isinstance(data, list):
        data = np.array(data)
        
    n_annotators = data.shape[1]
    
    if annotator_names is None:
        annotator_names = [f'Annotator_{i+1}' for i in range(n_annotators)]
    
    if len(annotator_names) != n_annotators:
        raise ValueError("Number of annotator names must match number of columns in data")
    
    # Create empty DataFrame for results
    results = pd.DataFrame(index=annotator_names, columns=annotator_names)
    
    # Calculate pairwise agreement for each pair of annotators
    for i, j in combinations(range(n_annotators), 2):
        # Extract data for just these two annotators
        pair_data = data[:, [i, j]]
        
        # Calculate agreement
        alpha = krippendorffs_alpha(pair_data, metric=metric)
        
        # Store in both positions of the symmetric matrix
        results.iloc[i, j] = alpha
        results.iloc[j, i] = alpha
    
    # Fill diagonal with 1.0
    np.fill_diagonal(results.values, 1.0)
    
    return results

def format_agreement_report(agreement_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the agreement report for better readability.
    
    Parameters:
    -----------
    agreement_df : pd.DataFrame
        Output from generate_pairwise_agreement_report
        
    Returns:
    --------
    pd.DataFrame
        Formatted agreement report with percentages
    """
    formatted_df = agreement_df.round(3) * 100
    return formatted_df.style.format("{:.1f}%")


