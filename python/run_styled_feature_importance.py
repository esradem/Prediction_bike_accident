#!/usr/bin/env python3
"""
Script to run styled feature importance analysis for bike accident hotspot prediction
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add notebook directory to path to import modules
sys.path.append('notebook')

# Import the styled feature importance functions
from styled_feature_importance import (
    train_and_plot_models, 
    style_temporal_plot,
    plot_feature_importance,
    plot_comparison_feature_importance
)

# Import required modules for clustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import hdbscan

class BikeAccidentHotspotPredictorLocal:
    """Local version with corrected data paths"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.coordinate_scaler = MinMaxScaler()
        self.clusterer = None
        self.df_coordinates = None
        self.df_weather = None
        self.df_injury_weekday = None
        self.hotspot_clusters = None
        
    def load_data(self):
        """Load the three datasets without merging them"""
        print("Loading datasets...")
        
        # Load coordinate data
        self.df_coordinates = pd.read_csv('data/cleaned/df_ams_cc.csv')
        print(f"Coordinates data shape: {self.df_coordinates.shape}")
        
        # Load weather data
        self.df_weather = pd.read_csv('data/cleaned/yearly_wea.csv')
        print(f"Weather data shape: {self.df_weather.shape}")
        
        # Load injury severity and weekday data
        self.df_injury_weekday = pd.read_csv('data/cleaned/gdf_joined.csv')
        print(f"Injury/Weekday data shape: {self.df_injury_weekday.shape}")
        
        return self
    
    def preprocess_coordinate_data(self):
        """Preprocess coordinate data for clustering"""
        print("Preprocessing coordinate data...")
        
        # Clean coordinate data
        coord_data = self.df_coordinates.copy()
        
        # Remove rows with missing coordinates
        coord_data = coord_data.dropna(subset=['longitude', 'latitude'])
        
        # Filter for Amsterdam area (reasonable coordinate bounds)
        coord_data = coord_data[
            (coord_data['longitude'].between(4.7, 5.1)) & 
            (coord_data['latitude'].between(52.2, 52.5))
        ]
        
        print(f"Cleaned coordinate data shape: {coord_data.shape}")
        
        # Create features for clustering
        features = []
        
        # 1. Spatial coordinates (primary clustering feature)
        coordinates = coord_data[['longitude', 'latitude']].values
        coordinates_scaled = self.coordinate_scaler.fit_transform(coordinates)
        features.append(coordinates_scaled)
        
        # 2. Temporal features from accident year
        if 'accident_year' in coord_data.columns:
            years = coord_data['accident_year'].fillna(coord_data['accident_year'].median())
            year_features = np.column_stack([
                years,
                np.sin(2 * np.pi * years / 7),  # Cyclical year pattern
                np.cos(2 * np.pi * years / 7)
            ])
            features.append(self.scaler.fit_transform(year_features))
        
        # Combine all features
        self.clustering_features = np.hstack(features)
        self.coord_data_clean = coord_data
        
        return self
    
    def apply_hdbscan_clustering(self, min_cluster_size=50, min_samples=10):
        """Apply HDBSCAN clustering to identify hotspots"""
        print("Applying HDBSCAN clustering...")
        
        # Initialize HDBSCAN with optimized parameters for hotspot detection
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_method='eom',
            alpha=1.0
        )
        
        # Fit the clustering model
        cluster_labels = self.clusterer.fit_predict(self.clustering_features)
        
        # Add cluster labels to coordinate data
        self.coord_data_clean['cluster'] = cluster_labels
        
        # Identify hotspot clusters (exclude noise points labeled as -1)
        unique_clusters = np.unique(cluster_labels)
        hotspot_clusters = unique_clusters[unique_clusters != -1]
        
        print(f"Found {len(hotspot_clusters)} hotspot clusters")
        print(f"Noise points: {np.sum(cluster_labels == -1)}")
        
        # Calculate silhouette score (excluding noise points)
        if len(hotspot_clusters) > 1:
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(
                    self.clustering_features[non_noise_mask], 
                    cluster_labels[non_noise_mask]
                )
                print(f"Silhouette Score: {silhouette_avg:.3f}")
        
        self.hotspot_clusters = hotspot_clusters
        return self

def main():
    """Main function to run styled feature importance analysis"""
    
    print("🚀 Starting Styled Feature Importance Analysis")
    print("=" * 50)
    
    try:
        # Initialize predictor and load data
        print("📊 Loading data and running clustering analysis...")
        predictor = BikeAccidentHotspotPredictorLocal()
        predictor.load_data()
        predictor.preprocess_coordinate_data()
        predictor.apply_hdbscan_clustering()
        
        # Get the processed coordinate data with cluster labels
        df_ams_cc = predictor.coord_data_clean.copy()
        
        print(f"✅ Data loaded successfully!")
        print(f"   - Total accidents: {len(df_ams_cc):,}")
        print(f"   - Clusters found: {len(predictor.hotspot_clusters)}")
        print(f"   - Noise points: {(df_ams_cc['cluster'] == -1).sum():,}")
        
        # Check if we have the required columns
        required_columns = ['latitude', 'longitude', 'accident_year', 'cluster']
        missing_columns = [col for col in required_columns if col not in df_ams_cc.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return
        
        print("\n📈 Creating styled feature importance plots...")
        
        # 1. Train models and create feature importance plots
        print("   🤖 Training Random Forest and XGBoost models...")
        rf_model, xgb_model = train_and_plot_models(df_ams_cc)
        
        print("   ✅ Feature importance plots created!")
        
        # 2. Create temporal trend plot
        print("   📅 Creating temporal trend analysis...")
        style_temporal_plot(df_ams_cc)
        
        print("   ✅ Temporal trend plot created!")
        
        # 3. Additional analysis - cluster distribution
        print("\n📊 Additional Analysis:")
        
        # Cluster statistics
        cluster_stats = df_ams_cc['cluster'].value_counts().sort_index()
        non_noise_clusters = cluster_stats[cluster_stats.index != -1]
        if len(non_noise_clusters) > 0:
            print(f"   - Largest cluster: {non_noise_clusters.iloc[0]} accidents (Cluster {non_noise_clusters.index[0]})")
            print(f"   - Average cluster size: {non_noise_clusters.mean():.1f} accidents")
        
        # Year range analysis
        year_range = df_ams_cc['accident_year'].agg(['min', 'max'])
        print(f"   - Data spans: {year_range['min']:.0f} - {year_range['max']:.0f}")
        
        # Geographic coverage
        lat_range = df_ams_cc['latitude'].agg(['min', 'max'])
        lon_range = df_ams_cc['longitude'].agg(['min', 'max'])
        print(f"   - Latitude range: {lat_range['min']:.4f} - {lat_range['max']:.4f}")
        print(f"   - Longitude range: {lon_range['min']:.4f} - {lon_range['max']:.4f}")
        
        print("\n🎯 Analysis completed successfully!")
        print("📊 Generated visualizations:")
        print("   - Random Forest feature importance plot")
        print("   - XGBoost feature importance plot") 
        print("   - Model comparison plot")
        print("   - Temporal trend analysis")
        
        return rf_model, xgb_model, df_ams_cc
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    rf_model, xgb_model, df_ams_cc = main() 