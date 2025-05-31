"""
Weather merge utilities for combining accident data with yearly weather data.

This module provides functions to merge accident datasets with weather data
based on the year column.
"""

import pandas as pd
import numpy as np
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

def load_accident_and_weather_data():
    """
    Load accident data and yearly weather data.
    
    Returns:
    tuple: (accident_df, weather_df)
    """
    data_dir = _find_data_directory()
    
    # Try to load accident data (prefer merged data with coordinates)
    accident_files = [
        'merged_ams_cc_accidents_matches_only.csv',
        'merged_ams_cc_accidents.csv',
        'df_ams_cc.csv'
    ]
    
    accident_df = None
    for filename in accident_files:
        file_path = os.path.join(data_dir, 'cleaned', filename)
        if os.path.exists(file_path):
            print(f"Loading accident data from: {filename}")
            accident_df = pd.read_csv(file_path)
            break
    
    if accident_df is None:
        raise FileNotFoundError("Could not find any accident data files")
    
    # Load weather data
    weather_file = os.path.join(data_dir, 'cleaned', 'yearly_wea.csv')
    if not os.path.exists(weather_file):
        raise FileNotFoundError(f"Weather data file not found: {weather_file}")
    
    print(f"Loading weather data from: yearly_wea.csv")
    weather_df = pd.read_csv(weather_file)
    
    return accident_df, weather_df

def merge_accident_weather_data(accident_df=None, weather_df=None, verbose=True):
    """
    Merge accident data with yearly weather data based on year.
    
    Parameters:
    accident_df (DataFrame, optional): Accident data. If None, will load automatically
    weather_df (DataFrame, optional): Weather data. If None, will load automatically
    verbose (bool): Whether to print detailed information
    
    Returns:
    DataFrame: Merged dataset with weather information
    """
    # Load data if not provided
    if accident_df is None or weather_df is None:
        if verbose:
            print("Loading datasets...")
        accident_df, weather_df = load_accident_and_weather_data()
    
    if verbose:
        print(f"\nDataset shapes before merge:")
        print(f"Accident data: {accident_df.shape}")
        print(f"Weather data: {weather_df.shape}")
    
    # Check for year columns
    if 'accident_year' not in accident_df.columns:
        raise ValueError("Accident data must have 'accident_year' column")
    
    if 'accident_year' not in weather_df.columns:
        raise ValueError("Weather data must have 'accident_year' column")
    
    # Clean year data
    accident_df['accident_year'] = pd.to_numeric(accident_df['accident_year'], errors='coerce')
    weather_df['accident_year'] = pd.to_numeric(weather_df['accident_year'], errors='coerce')
    
    # Remove rows with invalid years
    accident_df = accident_df.dropna(subset=['accident_year'])
    weather_df = weather_df.dropna(subset=['accident_year'])
    
    if verbose:
        print(f"\nYear ranges:")
        print(f"Accident data: {accident_df['accident_year'].min():.0f} - {accident_df['accident_year'].max():.0f}")
        print(f"Weather data: {weather_df['accident_year'].min():.0f} - {weather_df['accident_year'].max():.0f}")
    
    # Perform the merge
    merged_df = pd.merge(
        accident_df,
        weather_df,
        on='accident_year',
        how='left'  # Keep all accident records, add weather where available
    )
    
    if verbose:
        print(f"\nMerge results:")
        print(f"Total accident records: {len(accident_df)}")
        print(f"Records with weather data: {merged_df['temp_C'].notna().sum()}")
        print(f"Records without weather data: {merged_df['temp_C'].isna().sum()}")
        print(f"Final merged dataset shape: {merged_df.shape}")
    
    return merged_df

def analyze_weather_merge_results(merged_df, verbose=True):
    """
    Analyze the results of the weather merge.
    
    Parameters:
    merged_df (DataFrame): Merged dataset
    verbose (bool): Whether to print detailed analysis
    
    Returns:
    dict: Analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['total_records'] = len(merged_df)
    analysis['records_with_weather'] = merged_df['temp_C'].notna().sum()
    analysis['records_without_weather'] = merged_df['temp_C'].isna().sum()
    analysis['weather_coverage_pct'] = (analysis['records_with_weather'] / analysis['total_records']) * 100
    
    # Year-wise breakdown
    year_stats = merged_df.groupby('accident_year').agg({
        'temp_C': ['count', lambda x: x.notna().sum()],
        'accident_id_left': 'count'  # or any other accident column
    }).round(2)
    
    year_stats.columns = ['total_accidents', 'with_weather', 'accident_count']
    year_stats['weather_coverage_pct'] = (year_stats['with_weather'] / year_stats['total_accidents'] * 100).round(1)
    
    analysis['year_breakdown'] = year_stats
    
    # Weather statistics (for records with weather data)
    weather_subset = merged_df.dropna(subset=['temp_C'])
    if len(weather_subset) > 0:
        analysis['weather_stats'] = {
            'avg_temp_C': weather_subset['temp_C'].mean(),
            'avg_precip_mm': weather_subset['precip_mm'].mean(),
            'avg_wind_ms': weather_subset['wind_avg_ms'].mean(),
            'avg_visibility': weather_subset['visibility_range'].mean()
        }
    
    if verbose:
        print(f"\n📊 Weather Merge Analysis:")
        print(f"{'='*50}")
        print(f"Total records: {analysis['total_records']:,}")
        print(f"Records with weather: {analysis['records_with_weather']:,} ({analysis['weather_coverage_pct']:.1f}%)")
        print(f"Records without weather: {analysis['records_without_weather']:,}")
        
        print(f"\n📅 Year-wise breakdown:")
        print(year_stats)
        
        if 'weather_stats' in analysis:
            print(f"\n🌤️ Average weather conditions:")
            print(f"Temperature: {analysis['weather_stats']['avg_temp_C']:.1f}°C")
            print(f"Precipitation: {analysis['weather_stats']['avg_precip_mm']:.1f}mm")
            print(f"Wind speed: {analysis['weather_stats']['avg_wind_ms']:.1f}m/s")
            print(f"Visibility: {analysis['weather_stats']['avg_visibility']:.0f}m")
    
    return analysis

def save_weather_merged_data(merged_df, output_dir=None):
    """
    Save the weather-merged dataset to CSV file.
    
    Parameters:
    merged_df (DataFrame): Merged dataset
    output_dir (str, optional): Output directory. If None, uses data/cleaned
    
    Returns:
    str: Path to saved file
    """
    if output_dir is None:
        data_dir = _find_data_directory()
        output_dir = os.path.join(data_dir, 'cleaned')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the merged dataset
    output_path = os.path.join(output_dir, 'accident_weather_merged.csv')
    merged_df.to_csv(output_path, index=False)
    
    print(f"✅ Weather-merged dataset saved to: {output_path}")
    print(f"📁 File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    
    return output_path

def quick_weather_merge(save_results=True, verbose=True):
    """
    Perform a complete weather merge workflow.
    
    Parameters:
    save_results (bool): Whether to save the merged dataset
    verbose (bool): Whether to print detailed information
    
    Returns:
    tuple: (merged_df, analysis, file_path)
    """
    if verbose:
        print("🌦️ Starting weather merge workflow...")
        print("="*50)
    
    # Load and merge data
    merged_df = merge_accident_weather_data(verbose=verbose)
    
    # Analyze results
    analysis = analyze_weather_merge_results(merged_df, verbose=verbose)
    
    # Save results
    file_path = None
    if save_results:
        file_path = save_weather_merged_data(merged_df)
    
    if verbose:
        print(f"\n✅ Weather merge workflow completed!")
        if file_path:
            print(f"📁 Results saved to: {file_path}")
    
    return merged_df, analysis, file_path

def create_weather_visualization(merged_df):
    """
    Create visualizations of the weather-merged data.
    
    Parameters:
    merged_df (DataFrame): Merged dataset with weather data
    
    Returns:
    None (displays plots)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the plotting style
    plt.style.use('default')
    colors = {
        'primary': '#A8CEF1',
        'secondary': '#4B9FE1', 
        'background': '#F5F9FF',
        'text': '#2C3E50'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weather Conditions and Bike Accidents in Amsterdam', 
                 fontsize=16, fontweight='bold', color=colors['text'])
    
    # Filter data with weather information
    weather_data = merged_df.dropna(subset=['temp_C'])
    
    # 1. Accidents by year with weather data
    yearly_counts = weather_data.groupby('accident_year').size()
    axes[0,0].bar(yearly_counts.index, yearly_counts.values, 
                  color=colors['primary'], edgecolor=colors['secondary'])
    axes[0,0].set_title('Accidents by Year (with Weather Data)', fontweight='bold')
    axes[0,0].set_xlabel('Year')
    axes[0,0].set_ylabel('Number of Accidents')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Temperature distribution
    axes[0,1].hist(weather_data['temp_C'], bins=20, 
                   color=colors['primary'], edgecolor=colors['secondary'], alpha=0.7)
    axes[0,1].set_title('Temperature Distribution', fontweight='bold')
    axes[0,1].set_xlabel('Temperature (°C)')
    axes[0,1].set_ylabel('Number of Accidents')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Precipitation vs Accidents
    axes[1,0].scatter(weather_data['precip_mm'], weather_data['temp_C'], 
                      color=colors['secondary'], alpha=0.6)
    axes[1,0].set_title('Temperature vs Precipitation', fontweight='bold')
    axes[1,0].set_xlabel('Precipitation (mm)')
    axes[1,0].set_ylabel('Temperature (°C)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Wind speed distribution
    axes[1,1].hist(weather_data['wind_avg_ms'], bins=20, 
                   color=colors['primary'], edgecolor=colors['secondary'], alpha=0.7)
    axes[1,1].set_title('Wind Speed Distribution', fontweight='bold')
    axes[1,1].set_xlabel('Wind Speed (m/s)')
    axes[1,1].set_ylabel('Number of Accidents')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print(f"\n📈 Weather Data Summary:")
    print(f"Temperature: {weather_data['temp_C'].mean():.1f}°C ± {weather_data['temp_C'].std():.1f}")
    print(f"Precipitation: {weather_data['precip_mm'].mean():.1f}mm ± {weather_data['precip_mm'].std():.1f}")
    print(f"Wind Speed: {weather_data['wind_avg_ms'].mean():.1f}m/s ± {weather_data['wind_avg_ms'].std():.1f}")
    print(f"Visibility: {weather_data['visibility_range'].mean():.0f}m ± {weather_data['visibility_range'].std():.0f}")

if __name__ == "__main__":
    # Example usage
    print("Weather Merge Utilities - Example Usage")
    print("="*50)
    
    # Perform quick merge
    merged_data, analysis, file_path = quick_weather_merge()
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_weather_visualization(merged_data) 