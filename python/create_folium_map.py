"""
Create an interactive Folium map showing bike accident locations in Amsterdam.
This script loads the merged coordinate data and creates visualizations.
"""

import pandas as pd
import folium
from folium import plugins
import numpy as np
import os

def load_coordinate_data():
    """Load the coordinate data from available sources."""
    
    # Try to load merged data first (if available)
    possible_files = [
        'data/cleaned/merged_ams_cc_accidents_matches_only.csv',
        'data/cleaned/merged_ams_cc_accidents.csv',
        'data/cleaned/df_ams_cc.csv',
        'data/cleaned/gdf_joined.csv'
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Loading data from: {file_path}")
            df = pd.read_csv(file_path)
            
            # Check if we have the required columns
            if 'longitude' in df.columns and 'latitude' in df.columns:
                # Clean the data
                df_clean = df.dropna(subset=['longitude', 'latitude'])
                print(f"Loaded {len(df_clean)} records with valid coordinates")
                return df_clean
    
    raise FileNotFoundError("Could not find any data files with longitude/latitude coordinates")

def create_basic_map(df, map_title="Amsterdam Bike Accidents"):
    """Create a basic Folium map with accident locations."""
    
    # Calculate center of Amsterdam (approximate)
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    print(f"Map center: {center_lat:.4f}, {center_lon:.4f}")
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = f'''
                 <h3 align="center" style="font-size:20px"><b>{map_title}</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add markers for each accident (sample if too many)
    sample_size = min(1000, len(df))  # Limit to 1000 points for performance
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Sampling {sample_size} points from {len(df)} total points for performance")
    else:
        df_sample = df
    
    # Add markers
    for idx, row in df_sample.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=3,
            popup=f"Accident at ({row['latitude']:.4f}, {row['longitude']:.4f})",
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.6
        ).add_to(m)
    
    return m

def create_heatmap(df, map_title="Amsterdam Bike Accidents - Heatmap"):
    """Create a heatmap visualization of accident locations."""
    
    # Calculate center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = f'''
                 <h3 align="center" style="font-size:20px"><b>{map_title}</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Prepare data for heatmap
    heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows()]
    
    # Add heatmap
    plugins.HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    return m

def create_cluster_map(df, map_title="Amsterdam Bike Accidents - Clustered"):
    """Create a map with clustered markers."""
    
    # Calculate center
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Add title
    title_html = f'''
                 <h3 align="center" style="font-size:20px"><b>{map_title}</b></h3>
                 '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Create marker cluster
    marker_cluster = plugins.MarkerCluster().add_to(m)
    
    # Add markers to cluster
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=f"Accident at ({row['latitude']:.4f}, {row['longitude']:.4f})",
            icon=folium.Icon(color='red', icon='exclamation-sign')
        ).add_to(marker_cluster)
    
    return m

def create_all_maps(save_to_notebook_dir=True):
    """Create all map types and save them."""
    
    # Load data
    print("Loading coordinate data...")
    df = load_coordinate_data()
    
    # Determine output directory
    if save_to_notebook_dir and os.path.exists('notebook'):
        output_dir = 'notebook'
    else:
        output_dir = '.'
    
    maps_created = []
    
    # Create basic map
    print("\nCreating basic map...")
    basic_map = create_basic_map(df)
    basic_path = os.path.join(output_dir, 'amsterdam_accidents_basic.html')
    basic_map.save(basic_path)
    maps_created.append(('Basic Map', basic_path))
    print(f"✅ Basic map saved to: {basic_path}")
    
    # Create heatmap
    print("\nCreating heatmap...")
    heat_map = create_heatmap(df)
    heat_path = os.path.join(output_dir, 'amsterdam_accidents_heatmap.html')
    heat_map.save(heat_path)
    maps_created.append(('Heatmap', heat_path))
    print(f"✅ Heatmap saved to: {heat_path}")
    
    # Create cluster map
    print("\nCreating cluster map...")
    cluster_map = create_cluster_map(df)
    cluster_path = os.path.join(output_dir, 'amsterdam_accidents_clustered.html')
    cluster_map.save(cluster_path)
    maps_created.append(('Cluster Map', cluster_path))
    print(f"✅ Cluster map saved to: {cluster_path}")
    
    # Summary
    print(f"\n🗺️ Created {len(maps_created)} interactive maps:")
    for map_name, path in maps_created:
        print(f"   {map_name}: {path}")
    
    print(f"\n📊 Dataset summary:")
    print(f"   Total records: {len(df)}")
    print(f"   Latitude range: {df['latitude'].min():.4f} to {df['latitude'].max():.4f}")
    print(f"   Longitude range: {df['longitude'].min():.4f} to {df['longitude'].max():.4f}")
    
    return maps_created, df

def create_notebook_friendly_map(df=None):
    """Create a map that can be easily displayed in Jupyter notebooks."""
    
    if df is None:
        df = load_coordinate_data()
    
    # Create a compact heatmap for notebook display
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles='OpenStreetMap',
        width='100%',
        height='500px'
    )
    
    # Add heatmap
    heat_data = [[row['latitude'], row['longitude']] for idx, row in df.iterrows()]
    plugins.HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m)
    
    return m

if __name__ == "__main__":
    # Create all maps
    maps, data = create_all_maps()
    
    print("\n🎯 To view the maps:")
    print("   1. Open any of the HTML files in your web browser")
    print("   2. Or use the create_notebook_friendly_map() function in Jupyter") 