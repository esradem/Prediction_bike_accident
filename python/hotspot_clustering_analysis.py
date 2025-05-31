"""
Hotspot Clustering Analysis for Bike Accidents in Amsterdam using HDBSCAN.

This module provides functions to perform spatial clustering analysis on bike accident
data to identify hotspots and high-risk areas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import geopandas as gpd
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import hdbscan
import os
import warnings
warnings.filterwarnings('ignore')

def _find_data_directory():
    """Find the data directory relative to current working directory."""
    possible_data_paths = ['data', '../data', '../../data']
    for path in possible_data_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path
    return 'data'

def load_merged_data():
    """
    Load the merged accident data with coordinates and weather information.
    
    Returns:
    DataFrame: Merged dataset with coordinates
    """
    data_dir = _find_data_directory()
    
    # Try to load merged data files in order of preference
    possible_files = [
        'accident_weather_merged.csv',
        'merged_ams_cc_accidents_matches_only.csv',
        'merged_ams_cc_accidents.csv',
        'df_ams_cc.csv'
    ]
    
    for filename in possible_files:
        file_path = os.path.join(data_dir, 'cleaned', filename)
        if os.path.exists(file_path):
            print(f"Loading data from: {filename}")
            df = pd.read_csv(file_path)
            
            # Check if we have coordinate columns
            coord_cols = ['longitude', 'latitude']
            if all(col in df.columns for col in coord_cols):
                # Remove rows with missing coordinates
                df = df.dropna(subset=coord_cols)
                print(f"Loaded {len(df)} records with coordinates")
                return df
            else:
                print(f"File {filename} doesn't have coordinate columns")
                continue
    
    raise FileNotFoundError("Could not find merged data with coordinates")

def prepare_clustering_data(df, include_weather=True, include_temporal=True):
    """
    Prepare data for clustering analysis.
    
    Parameters:
    df (DataFrame): Input dataset
    include_weather (bool): Whether to include weather features
    include_temporal (bool): Whether to include temporal features
    
    Returns:
    tuple: (features_df, feature_names)
    """
    # Start with coordinates
    features = df[['longitude', 'latitude']].copy()
    feature_names = ['longitude', 'latitude']
    
    # Add temporal features if available and requested
    if include_temporal and 'accident_year' in df.columns:
        features['accident_year'] = df['accident_year']
        feature_names.append('accident_year')
    
    if include_temporal and 'accident_month' in df.columns:
        features['accident_month'] = df['accident_month']
        feature_names.append('accident_month')
    
    # Add weather features if available and requested
    if include_weather:
        weather_cols = ['temp_C', 'precip_mm', 'wind_avg_ms', 'visibility_range']
        available_weather = [col for col in weather_cols if col in df.columns]
        
        if available_weather:
            for col in available_weather:
                # Fill missing weather data with median
                features[col] = df[col].fillna(df[col].median())
                feature_names.append(col)
            print(f"Added weather features: {available_weather}")
    
    # Remove any remaining NaN values
    features = features.dropna()
    
    print(f"Prepared {len(features)} records with {len(feature_names)} features")
    print(f"Features: {feature_names}")
    
    return features, feature_names

def perform_hdbscan_clustering(features, min_cluster_size=10, min_samples=5, 
                              metric='euclidean', cluster_selection_epsilon=0.0):
    """
    Perform HDBSCAN clustering on the features.
    
    Parameters:
    features (DataFrame): Feature matrix
    min_cluster_size (int): Minimum cluster size
    min_samples (int): Minimum samples in a cluster
    metric (str): Distance metric
    cluster_selection_epsilon (float): Distance threshold for cluster selection
    
    Returns:
    tuple: (clusterer, labels, probabilities)
    """
    print(f"\n🔍 Performing HDBSCAN clustering...")
    print(f"Parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}")
    
    # Standardize features for clustering
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_epsilon=cluster_selection_epsilon
    )
    
    cluster_labels = clusterer.fit_predict(features_scaled)
    
    # Get cluster probabilities
    probabilities = clusterer.probabilities_
    
    # Analyze results
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"\n📊 Clustering Results:")
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise}")
    print(f"Percentage of data in clusters: {((len(cluster_labels) - n_noise) / len(cluster_labels) * 100):.1f}%")
    
    if n_clusters > 1:
        # Calculate silhouette score (excluding noise points)
        mask = cluster_labels != -1
        if np.sum(mask) > 1:
            silhouette_avg = silhouette_score(features_scaled[mask], cluster_labels[mask])
            print(f"Average silhouette score: {silhouette_avg:.3f}")
    
    return clusterer, cluster_labels, probabilities, scaler

def analyze_clusters(df, cluster_labels, probabilities):
    """
    Analyze the characteristics of identified clusters.
    
    Parameters:
    df (DataFrame): Original dataset
    cluster_labels (array): Cluster labels
    probabilities (array): Cluster membership probabilities
    
    Returns:
    DataFrame: Cluster analysis results
    """
    # Add cluster information to dataframe
    df_clustered = df.copy()
    df_clustered['cluster'] = cluster_labels
    df_clustered['cluster_probability'] = probabilities
    
    # Analyze clusters
    cluster_stats = []
    
    for cluster_id in sorted(set(cluster_labels)):
        if cluster_id == -1:
            cluster_name = "Noise"
        else:
            cluster_name = f"Cluster {cluster_id}"
        
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        stats = {
            'cluster_id': cluster_id,
            'cluster_name': cluster_name,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(df_clustered) * 100,
            'avg_longitude': cluster_data['longitude'].mean(),
            'avg_latitude': cluster_data['latitude'].mean(),
            'avg_probability': cluster_data['cluster_probability'].mean()
        }
        
        # Add weather statistics if available
        if 'temp_C' in cluster_data.columns:
            stats['avg_temp_C'] = cluster_data['temp_C'].mean()
        if 'precip_mm' in cluster_data.columns:
            stats['avg_precip_mm'] = cluster_data['precip_mm'].mean()
        if 'wind_avg_ms' in cluster_data.columns:
            stats['avg_wind_ms'] = cluster_data['wind_avg_ms'].mean()
        
        # Add temporal statistics if available
        if 'accident_year' in cluster_data.columns:
            stats['year_range'] = f"{cluster_data['accident_year'].min():.0f}-{cluster_data['accident_year'].max():.0f}"
        
        cluster_stats.append(stats)
    
    cluster_analysis = pd.DataFrame(cluster_stats)
    
    print(f"\n📈 Cluster Analysis:")
    print("="*80)
    for _, row in cluster_analysis.iterrows():
        if row['cluster_id'] != -1:
            print(f"\n{row['cluster_name']}:")
            print(f"  Size: {row['size']} accidents ({row['percentage']:.1f}%)")
            print(f"  Center: ({row['avg_longitude']:.4f}, {row['avg_latitude']:.4f})")
            print(f"  Avg Probability: {row['avg_probability']:.3f}")
            if 'avg_temp_C' in row:
                print(f"  Avg Temperature: {row['avg_temp_C']:.1f}°C")
            if 'year_range' in row:
                print(f"  Year Range: {row['year_range']}")
    
    return df_clustered, cluster_analysis

def create_cluster_visualizations(df_clustered, cluster_analysis):
    """
    Create visualizations of the clustering results.
    
    Parameters:
    df_clustered (DataFrame): Dataset with cluster labels
    cluster_analysis (DataFrame): Cluster analysis results
    """
    # Set up plotting style
    plt.style.use('default')
    colors = {
        'primary': '#A8CEF1',
        'secondary': '#4B9FE1', 
        'background': '#F5F9FF',
        'text': '#2C3E50'
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('HDBSCAN Hotspot Clustering Analysis - Amsterdam Bike Accidents', 
                 fontsize=16, fontweight='bold', color=colors['text'])
    
    # 1. Scatter plot of clusters
    clusters = df_clustered[df_clustered['cluster'] != -1]
    noise = df_clustered[df_clustered['cluster'] == -1]
    
    # Plot noise points
    if len(noise) > 0:
        axes[0,0].scatter(noise['longitude'], noise['latitude'], 
                         c='lightgray', alpha=0.5, s=10, label='Noise')
    
    # Plot clusters with different colors
    if len(clusters) > 0:
        scatter = axes[0,0].scatter(clusters['longitude'], clusters['latitude'], 
                                   c=clusters['cluster'], cmap='tab10', 
                                   alpha=0.7, s=20)
        plt.colorbar(scatter, ax=axes[0,0], label='Cluster ID')
    
    axes[0,0].set_title('Spatial Distribution of Clusters', fontweight='bold')
    axes[0,0].set_xlabel('Longitude')
    axes[0,0].set_ylabel('Latitude')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Cluster sizes
    cluster_sizes = cluster_analysis[cluster_analysis['cluster_id'] != -1]
    if len(cluster_sizes) > 0:
        bars = axes[0,1].bar(cluster_sizes['cluster_name'], cluster_sizes['size'],
                            color=colors['primary'], edgecolor=colors['secondary'])
        axes[0,1].set_title('Cluster Sizes', fontweight='bold')
        axes[0,1].set_xlabel('Cluster')
        axes[0,1].set_ylabel('Number of Accidents')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{int(height)}', ha='center', va='bottom')
    
    # 3. Cluster probabilities
    if len(clusters) > 0:
        axes[1,0].hist(clusters['cluster_probability'], bins=20,
                      color=colors['primary'], edgecolor=colors['secondary'], alpha=0.7)
        axes[1,0].set_title('Cluster Membership Probabilities', fontweight='bold')
        axes[1,0].set_xlabel('Probability')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].grid(True, alpha=0.3)
    
    # 4. Weather conditions by cluster (if available)
    if 'temp_C' in df_clustered.columns and len(clusters) > 0:
        cluster_weather = clusters.groupby('cluster')['temp_C'].mean()
        bars = axes[1,1].bar(range(len(cluster_weather)), cluster_weather.values,
                            color=colors['primary'], edgecolor=colors['secondary'])
        axes[1,1].set_title('Average Temperature by Cluster', fontweight='bold')
        axes[1,1].set_xlabel('Cluster ID')
        axes[1,1].set_ylabel('Temperature (°C)')
        axes[1,1].set_xticks(range(len(cluster_weather)))
        axes[1,1].set_xticklabels(cluster_weather.index)
        axes[1,1].grid(True, alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}°C', ha='center', va='bottom')
    else:
        # If no weather data, show year distribution
        if 'accident_year' in df_clustered.columns and len(clusters) > 0:
            year_dist = clusters.groupby('cluster')['accident_year'].mean()
            axes[1,1].bar(range(len(year_dist)), year_dist.values,
                         color=colors['primary'], edgecolor=colors['secondary'])
            axes[1,1].set_title('Average Year by Cluster', fontweight='bold')
            axes[1,1].set_xlabel('Cluster ID')
            axes[1,1].set_ylabel('Average Year')
            axes[1,1].set_xticks(range(len(year_dist)))
            axes[1,1].set_xticklabels(year_dist.index)
            axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_interactive_cluster_map(df_clustered, cluster_analysis):
    """
    Create an interactive Folium map showing the clusters.
    
    Parameters:
    df_clustered (DataFrame): Dataset with cluster labels
    cluster_analysis (DataFrame): Cluster analysis results
    
    Returns:
    folium.Map: Interactive map
    """
    # Calculate map center
    center_lat = df_clustered['latitude'].mean()
    center_lon = df_clustered['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Define colors for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
              'gray', 'black', 'lightgray']
    
    # Add cluster points
    clusters = df_clustered[df_clustered['cluster'] != -1]
    noise = df_clustered[df_clustered['cluster'] == -1]
    
    # Add noise points
    if len(noise) > 0:
        for _, row in noise.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                popup=f"Noise Point<br>Probability: {row['cluster_probability']:.3f}",
                color='gray',
                fillColor='lightgray',
                fillOpacity=0.5
            ).add_to(m)
    
    # Add cluster points
    for cluster_id in sorted(clusters['cluster'].unique()):
        cluster_data = clusters[clusters['cluster'] == cluster_id]
        color = colors[cluster_id % len(colors)]
        
        for _, row in cluster_data.iterrows():
            popup_text = f"Cluster {cluster_id}<br>"
            popup_text += f"Probability: {row['cluster_probability']:.3f}<br>"
            if 'accident_year' in row:
                popup_text += f"Year: {row['accident_year']:.0f}<br>"
            if 'temp_C' in row and pd.notna(row['temp_C']):
                popup_text += f"Temperature: {row['temp_C']:.1f}°C<br>"
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                popup=popup_text,
                color=color,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
    
    # Add cluster centers
    cluster_centers = cluster_analysis[cluster_analysis['cluster_id'] != -1]
    for _, row in cluster_centers.iterrows():
        folium.Marker(
            location=[row['avg_latitude'], row['avg_longitude']],
            popup=f"Cluster {row['cluster_id']} Center<br>Size: {row['size']} accidents",
            icon=folium.Icon(color='black', icon='star')
        ).add_to(m)
    
    # Add heatmap layer
    if len(df_clustered) > 0:
        heat_data = [[row['latitude'], row['longitude']] for _, row in df_clustered.iterrows()]
        plugins.HeatMap(heat_data, name='Accident Heatmap').add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def save_clustering_results(df_clustered, cluster_analysis, output_dir=None):
    """
    Save clustering results to files.
    
    Parameters:
    df_clustered (DataFrame): Dataset with cluster labels
    cluster_analysis (DataFrame): Cluster analysis results
    output_dir (str, optional): Output directory
    
    Returns:
    dict: Saved file paths
    """
    if output_dir is None:
        data_dir = _find_data_directory()
        output_dir = os.path.join(data_dir, 'cleaned')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save clustered data
    clustered_path = os.path.join(output_dir, 'accident_hotspot_clusters.csv')
    df_clustered.to_csv(clustered_path, index=False)
    
    # Save cluster analysis
    analysis_path = os.path.join(output_dir, 'cluster_analysis_summary.csv')
    cluster_analysis.to_csv(analysis_path, index=False)
    
    print(f"\n✅ Clustering results saved:")
    print(f"📁 Clustered data: {clustered_path}")
    print(f"📁 Analysis summary: {analysis_path}")
    
    return {
        'clustered_data': clustered_path,
        'analysis_summary': analysis_path
    }

def quick_hotspot_analysis(min_cluster_size=10, min_samples=5, 
                          include_weather=True, include_temporal=True,
                          save_results=True, create_map=True):
    """
    Perform complete hotspot clustering analysis workflow.
    
    Parameters:
    min_cluster_size (int): Minimum cluster size for HDBSCAN
    min_samples (int): Minimum samples for HDBSCAN
    include_weather (bool): Include weather features
    include_temporal (bool): Include temporal features
    save_results (bool): Save results to files
    create_map (bool): Create interactive map
    
    Returns:
    tuple: (df_clustered, cluster_analysis, map_object, file_paths)
    """
    print("🎯 Starting Hotspot Clustering Analysis")
    print("="*60)
    
    # Load data
    df = load_merged_data()
    
    # Prepare features
    features, feature_names = prepare_clustering_data(
        df, include_weather=include_weather, include_temporal=include_temporal
    )
    
    # Perform clustering
    clusterer, labels, probabilities, scaler = perform_hdbscan_clustering(
        features, min_cluster_size=min_cluster_size, min_samples=min_samples
    )
    
    # Analyze clusters
    df_clustered, cluster_analysis = analyze_clusters(df, labels, probabilities)
    
    # Create visualizations
    create_cluster_visualizations(df_clustered, cluster_analysis)
    
    # Create interactive map
    map_obj = None
    if create_map:
        print("\n🗺️ Creating interactive map...")
        map_obj = create_interactive_cluster_map(df_clustered, cluster_analysis)
        
        # Save map
        data_dir = _find_data_directory()
        map_path = os.path.join(data_dir, 'cleaned', 'hotspot_clusters_map.html')
        map_obj.save(map_path)
        print(f"📁 Interactive map saved: {map_path}")
    
    # Save results
    file_paths = {}
    if save_results:
        file_paths = save_clustering_results(df_clustered, cluster_analysis)
    
    print(f"\n✅ Hotspot clustering analysis completed!")
    
    return df_clustered, cluster_analysis, map_obj, file_paths

if __name__ == "__main__":
    # Example usage
    print("HDBSCAN Hotspot Clustering Analysis")
    print("="*50)
    
    # Perform complete analysis
    clustered_data, analysis, map_obj, files = quick_hotspot_analysis(
        min_cluster_size=15,
        min_samples=10,
        include_weather=True,
        include_temporal=True
    ) 