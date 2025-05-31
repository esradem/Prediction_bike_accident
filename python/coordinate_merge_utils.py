"""
Coordinate-based merge utilities for df_ams_cc and df_accidents datasets.

This module provides functions to merge datasets based on longitude and latitude
coordinates using spatial proximity matching.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import os

def _find_data_directory():
    """
    Find the data directory relative to current working directory.
    
    Returns:
    str: Path to data directory
    """
    possible_data_paths = [
        'data',
        '../data',
        '../../data'
    ]
    
    for path in possible_data_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    
    return 'data'  # fallback

def _find_file_paths():
    """
    Find the correct paths for the required CSV files.
    
    Returns:
    tuple: (ams_path, accidents_path, data_dir)
    """
    # Try different possible paths
    possible_paths = [
        ('data/cleaned/df_ams_cc.csv', 'data/cleaned/gdf_joined.csv', 'data/cleaned'),
        ('../data/cleaned/df_ams_cc.csv', '../data/cleaned/gdf_joined.csv', '../data/cleaned'),
        ('../../data/cleaned/df_ams_cc.csv', '../../data/cleaned/gdf_joined.csv', '../../data/cleaned'),
    ]
    
    for ams_path, accidents_path, data_dir in possible_paths:
        if os.path.exists(ams_path) and os.path.exists(accidents_path):
            return ams_path, accidents_path, data_dir
    
    return None, None, None

def load_datasets():
    """
    Load the df_ams_cc and df_accidents datasets.
    
    Returns:
    tuple: (df_ams_cc, df_accidents)
    """
    ams_path, accidents_path, _ = _find_file_paths()
    
    if ams_path and accidents_path:
        print(f"Loading datasets from: {ams_path} and {accidents_path}")
        df_ams_cc = pd.read_csv(ams_path)
        df_accidents = pd.read_csv(accidents_path)
        return df_ams_cc, df_accidents
    
    # If no paths work, show available files and raise error
    print("❌ Could not find the required data files!")
    print(f"Current working directory: {os.getcwd()}")
    print("Looking for files:")
    print("  - df_ams_cc.csv")
    print("  - gdf_joined.csv")
    
    # Show what files are available
    for root, dirs, files in os.walk('.'):
        if 'df_ams_cc.csv' in files or 'gdf_joined.csv' in files:
            print(f"Found data files in: {root}")
            for file in files:
                if file.endswith('.csv'):
                    print(f"  - {file}")
    
    raise FileNotFoundError(
        "Could not find df_ams_cc.csv and gdf_joined.csv in expected locations. "
        "Please ensure the files are in data/cleaned/ directory."
    )

def merge_on_coordinates(df_ams_cc, df_accidents, tolerance_meters=50, verbose=True):
    """
    Merge df_ams_cc and df_accidents based on longitude and latitude proximity.
    
    Parameters:
    df_ams_cc: DataFrame with accident data containing longitude and latitude
    df_accidents: DataFrame with accident data containing longitude and latitude  
    tolerance_meters: Maximum distance in meters for considering a match (default: 50m)
    verbose: Whether to print progress information
    
    Returns:
    merged_df: DataFrame with merged data
    """
    
    if verbose:
        print("Starting coordinate-based merge...")
        print(f"df_ams_cc shape: {df_ams_cc.shape}")
        print(f"df_accidents shape: {df_accidents.shape}")
    
    # Clean and prepare the data
    df_ams_cc_clean = df_ams_cc.dropna(subset=['longitude', 'latitude']).copy()
    df_accidents_clean = df_accidents.dropna(subset=['longitude', 'latitude']).copy()
    
    # Remove conflicting columns that might interfere with spatial join
    conflicting_cols = ['index_right', 'index_left']
    for col in conflicting_cols:
        if col in df_ams_cc_clean.columns:
            df_ams_cc_clean = df_ams_cc_clean.drop(columns=[col])
        if col in df_accidents_clean.columns:
            df_accidents_clean = df_accidents_clean.drop(columns=[col])
    
    if verbose:
        print(f"After cleaning:")
        print(f"df_ams_cc_clean shape: {df_ams_cc_clean.shape}")
        print(f"df_accidents_clean shape: {df_accidents_clean.shape}")
    
    # Convert to GeoDataFrames for spatial operations
    gdf_ams_cc = gpd.GeoDataFrame(
        df_ams_cc_clean, 
        geometry=gpd.points_from_xy(df_ams_cc_clean.longitude, df_ams_cc_clean.latitude),
        crs='EPSG:4326'
    )
    
    gdf_accidents = gpd.GeoDataFrame(
        df_accidents_clean,
        geometry=gpd.points_from_xy(df_accidents_clean.longitude, df_accidents_clean.latitude), 
        crs='EPSG:4326'
    )
    
    # Convert to a projected CRS for accurate distance calculations (Dutch RD New)
    gdf_ams_cc_proj = gdf_ams_cc.to_crs('EPSG:28992')
    gdf_accidents_proj = gdf_accidents.to_crs('EPSG:28992')
    
    # Perform spatial join with distance tolerance
    if verbose:
        print(f"Performing spatial join with {tolerance_meters}m tolerance...")
    
    merged_gdf = gpd.sjoin_nearest(
        gdf_ams_cc_proj, 
        gdf_accidents_proj, 
        how='left',
        max_distance=tolerance_meters,
        distance_col='distance_m',
        rsuffix='_accidents'
    )
    
    # Convert back to regular DataFrame
    merged_df = pd.DataFrame(merged_gdf.drop(columns='geometry'))
    
    if verbose:
        print(f"Merged dataset shape: {merged_df.shape}")
        print(f"Successful matches: {merged_df['distance_m'].notna().sum()}")
        if merged_df['distance_m'].notna().sum() > 0:
            print(f"Average distance for matches: {merged_df['distance_m'].mean():.2f}m")
            print(f"Match rate: {(merged_df['distance_m'].notna().sum() / len(merged_df)) * 100:.1f}%")
    
    return merged_df

def get_matched_records_only(merged_df):
    """
    Extract only the records that had successful coordinate matches.
    
    Parameters:
    merged_df: DataFrame from merge_on_coordinates function
    
    Returns:
    DataFrame with only matched records
    """
    return merged_df[merged_df['distance_m'].notna()].copy()

def analyze_merge_results(merged_df):
    """
    Analyze the results of the coordinate-based merge.
    
    Parameters:
    merged_df: DataFrame from merge_on_coordinates function
    
    Returns:
    dict: Dictionary with analysis results
    """
    matches = get_matched_records_only(merged_df)
    
    analysis = {
        'total_records': len(merged_df),
        'matched_records': len(matches),
        'unmatched_records': len(merged_df) - len(matches),
        'match_rate_percent': (len(matches) / len(merged_df)) * 100 if len(merged_df) > 0 else 0,
        'distance_stats': {}
    }
    
    if len(matches) > 0:
        analysis['distance_stats'] = {
            'min_distance_m': matches['distance_m'].min(),
            'max_distance_m': matches['distance_m'].max(),
            'mean_distance_m': matches['distance_m'].mean(),
            'median_distance_m': matches['distance_m'].median(),
            'std_distance_m': matches['distance_m'].std()
        }
    
    return analysis

def save_merged_data(merged_df, output_path=None, save_matches_only=True):
    """
    Save the merged dataset to CSV file(s).
    
    Parameters:
    merged_df: DataFrame from merge_on_coordinates function
    output_path: Path for the main merged dataset (auto-detected if None)
    save_matches_only: Whether to also save a file with only matched records
    
    Returns:
    dict: Dictionary with file paths
    """
    # Auto-detect output path if not provided
    if output_path is None:
        _, _, data_dir = _find_file_paths()
        if data_dir:
            output_path = os.path.join(data_dir, 'merged_ams_cc_accidents.csv')
        else:
            # Fallback to current directory
            output_path = 'merged_ams_cc_accidents.csv'
    
    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")
    
    # Save full merged dataset
    merged_df.to_csv(output_path, index=False)
    files_saved = {'merged_data': output_path}
    
    # Save matches only if requested
    if save_matches_only:
        matches = get_matched_records_only(merged_df)
        if len(matches) > 0:
            matches_path = output_path.replace('.csv', '_matches_only.csv')
            matches.to_csv(matches_path, index=False)
            files_saved['matches_only'] = matches_path
    
    return files_saved

def quick_merge(tolerance_meters=50, save_results=True, verbose=True):
    """
    Convenience function to perform the complete merge workflow.
    
    Parameters:
    tolerance_meters: Distance tolerance for matching
    save_results: Whether to save the results to files
    verbose: Whether to print progress information
    
    Returns:
    tuple: (merged_df, analysis_results, file_paths)
    """
    # Load datasets
    if verbose:
        print("Loading datasets...")
    df_ams_cc, df_accidents = load_datasets()
    
    # Perform merge
    merged_df = merge_on_coordinates(df_ams_cc, df_accidents, tolerance_meters, verbose)
    
    # Analyze results
    analysis = analyze_merge_results(merged_df)
    
    # Save results if requested
    file_paths = {}
    if save_results:
        file_paths = save_merged_data(merged_df)
        if verbose:
            print(f"\nFiles saved:")
            for key, path in file_paths.items():
                print(f"  {key}: {path}")
    
    # Print summary
    if verbose:
        print(f"\n=== MERGE SUMMARY ===")
        print(f"Total records: {analysis['total_records']}")
        print(f"Matched records: {analysis['matched_records']}")
        print(f"Match rate: {analysis['match_rate_percent']:.1f}%")
        
        if analysis['distance_stats']:
            print(f"Average distance: {analysis['distance_stats']['mean_distance_m']:.2f}m")
            print(f"Distance range: {analysis['distance_stats']['min_distance_m']:.2f}m - {analysis['distance_stats']['max_distance_m']:.2f}m")
    
    return merged_df, analysis, file_paths

# Example usage:
if __name__ == "__main__":
    # Simple usage
    merged_data, results, files = quick_merge(tolerance_meters=50)
    
    # Or step by step
    # df_ams_cc, df_accidents = load_datasets()
    # merged_df = merge_on_coordinates(df_ams_cc, df_accidents, tolerance_meters=100)
    # analysis = analyze_merge_results(merged_df)
    # files = save_merged_data(merged_df) 