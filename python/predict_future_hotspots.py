#!/usr/bin/env python3
"""
Future Bike Accident Hotspot Prediction using HDBSCAN Clustering

This script predicts future bike accident hotspots by:
1. Using coordinates from df_ams_cc.csv for spatial clustering
2. Incorporating yearly weather data from yearly_wea.csv
3. Using injury severity and weekday patterns from gdf_joined.csv
4. Applying HDBSCAN clustering to identify current hotspots
5. Predicting future hotspots based on weather and temporal patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import hdbscan
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings
warnings.filterwarnings('ignore')

class BikeAccidentHotspotPredictor:
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
        self.df_coordinates = pd.read_csv('../data/cleaned/df_ams_cc.csv')
        print(f"Coordinates data shape: {self.df_coordinates.shape}")
        
        # Load weather data
        self.df_weather = pd.read_csv('../data/cleaned/yearly_wea.csv')
        print(f"Weather data shape: {self.df_weather.shape}")
        
        # Load injury severity and weekday data
        self.df_injury_weekday = pd.read_csv('../data/cleaned/gdf_joined.csv')
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
    
    def analyze_hotspot_characteristics(self):
        """Analyze characteristics of identified hotspots"""
        print("Analyzing hotspot characteristics...")
        
        hotspot_analysis = []
        
        for cluster_id in self.hotspot_clusters:
            cluster_data = self.coord_data_clean[self.coord_data_clean['cluster'] == cluster_id]
            
            analysis = {
                'cluster_id': cluster_id,
                'accident_count': len(cluster_data),
                'center_lat': cluster_data['latitude'].mean(),
                'center_lon': cluster_data['longitude'].mean(),
                'lat_std': cluster_data['latitude'].std(),
                'lon_std': cluster_data['longitude'].std(),
                'year_range': f"{cluster_data['accident_year'].min():.0f}-{cluster_data['accident_year'].max():.0f}",
                'avg_year': cluster_data['accident_year'].mean()
            }
            
            # Add street information if available
            if 'streetname' in cluster_data.columns:
                top_streets = cluster_data['streetname'].value_counts().head(3)
                analysis['top_streets'] = ', '.join([f"{street} ({count})" for street, count in top_streets.items()])
            
            hotspot_analysis.append(analysis)
        
        self.hotspot_analysis = pd.DataFrame(hotspot_analysis)
        self.hotspot_analysis = self.hotspot_analysis.sort_values('accident_count', ascending=False)
        
        print("\nTop 10 Hotspots by Accident Count:")
        print(self.hotspot_analysis.head(10)[['cluster_id', 'accident_count', 'center_lat', 'center_lon', 'top_streets']])
        
        return self
    
    def incorporate_weather_patterns(self):
        """Analyze weather patterns for prediction"""
        print("Analyzing weather patterns...")
        
        # Weather trends analysis
        weather_trends = self.df_weather.copy()
        
        # Calculate weather trend indicators
        weather_trends['temp_trend'] = weather_trends['temp_C'].pct_change()
        weather_trends['precip_trend'] = weather_trends['precip_mm'].pct_change()
        weather_trends['wind_trend'] = weather_trends['wind_avg_ms'].pct_change()
        weather_trends['visibility_trend'] = weather_trends['visibility_range'].pct_change()
        
        # Identify weather risk factors
        weather_trends['high_risk_weather'] = (
            (weather_trends['temp_C'] < 5) |  # Very cold
            (weather_trends['precip_mm'] > 3) |  # Heavy precipitation
            (weather_trends['wind_avg_ms'] > 6) |  # Strong wind
            (weather_trends['visibility_range'] < 20000)  # Poor visibility
        ).astype(int)
        
        self.weather_analysis = weather_trends
        
        print("Weather Risk Analysis:")
        print(weather_trends[['accident_year', 'total_accidents', 'high_risk_weather', 'temp_C', 'precip_mm']])
        
        return self
    
    def incorporate_injury_weekday_patterns(self):
        """Analyze injury severity and weekday patterns"""
        print("Analyzing injury severity and weekday patterns...")
        
        injury_data = self.df_injury_weekday.copy()
        
        # Calculate injury severity metrics
        injury_data['injury_rate'] = injury_data['injury_accidents'] / injury_data['total_accidents']
        injury_data['fatal_rate'] = injury_data['fatal_accidents'] / injury_data['total_accidents']
        injury_data['severity_score'] = (
            injury_data['fatal_accidents'] * 3 + 
            injury_data['injury_accidents'] * 2 + 
            injury_data['damage_only_accidents'] * 1
        ) / injury_data['total_accidents']
        
        # Analyze weekday patterns
        weekday_cols = [col for col in injury_data.columns if 'weekday_' in col]
        if weekday_cols:
            injury_data['avg_weekday_accidents'] = injury_data[weekday_cols].mean(axis=1)
            injury_data['weekday_variability'] = injury_data[weekday_cols].std(axis=1)
        
        # Identify high-risk locations
        injury_data['high_risk_location'] = (
            (injury_data['injury_rate'] > injury_data['injury_rate'].quantile(0.75)) |
            (injury_data['fatal_rate'] > 0) |
            (injury_data['severity_score'] > injury_data['severity_score'].quantile(0.8))
        ).astype(int)
        
        self.injury_analysis = injury_data
        
        print("Injury Severity Analysis:")
        print(f"High injury rate locations: {injury_data['high_risk_location'].sum()}")
        print(f"Average injury rate: {injury_data['injury_rate'].mean():.3f}")
        print(f"Locations with fatalities: {(injury_data['fatal_rate'] > 0).sum()}")
        
        return self
    
    def predict_future_hotspots(self, future_year=2025, weather_scenario='normal'):
        """Predict future hotspots based on patterns"""
        print(f"Predicting future hotspots for {future_year}...")
        
        # Define weather scenarios
        weather_scenarios = {
            'normal': {'temp_factor': 1.0, 'precip_factor': 1.0, 'wind_factor': 1.0},
            'harsh': {'temp_factor': 0.9, 'precip_factor': 1.3, 'wind_factor': 1.2},
            'mild': {'temp_factor': 1.1, 'precip_factor': 0.8, 'wind_factor': 0.9}
        }
        
        scenario = weather_scenarios.get(weather_scenario, weather_scenarios['normal'])
        
        # Calculate prediction scores for each hotspot
        predictions = []
        
        for _, hotspot in self.hotspot_analysis.iterrows():
            cluster_id = hotspot['cluster_id']
            
            # Base risk from historical accident count
            base_risk = np.log1p(hotspot['accident_count']) / 10
            
            # Weather risk factor
            latest_weather = self.weather_analysis.iloc[-1]
            weather_risk = (
                (1 - min(latest_weather['temp_C'] * scenario['temp_factor'] / 15, 1)) * 0.3 +
                (min(latest_weather['precip_mm'] * scenario['precip_factor'] / 5, 1)) * 0.4 +
                (min(latest_weather['wind_avg_ms'] * scenario['wind_factor'] / 8, 1)) * 0.2 +
                (1 - min(latest_weather['visibility_range'] / 25000, 1)) * 0.1
            )
            
            # Temporal trend (recent years weighted more)
            year_weight = max(0, (hotspot['avg_year'] - 2015) / 10)
            
            # Find nearby high-risk injury locations
            hotspot_coords = (hotspot['center_lat'], hotspot['center_lon'])
            nearby_injury_risk = 0
            
            for _, injury_loc in self.injury_analysis.iterrows():
                if injury_loc['high_risk_location']:
                    distance = np.sqrt(
                        (hotspot_coords[0] - injury_loc['latitude'])**2 + 
                        (hotspot_coords[1] - injury_loc['longitude'])**2
                    )
                    if distance < 0.01:  # Approximately 1km
                        nearby_injury_risk += injury_loc['severity_score']
            
            nearby_injury_risk = min(nearby_injury_risk / 5, 1)  # Normalize
            
            # Calculate final prediction score
            prediction_score = (
                base_risk * 0.4 +
                weather_risk * 0.3 +
                year_weight * 0.2 +
                nearby_injury_risk * 0.1
            )
            
            predictions.append({
                'cluster_id': cluster_id,
                'center_lat': hotspot['center_lat'],
                'center_lon': hotspot['center_lon'],
                'historical_accidents': hotspot['accident_count'],
                'base_risk': base_risk,
                'weather_risk': weather_risk,
                'temporal_weight': year_weight,
                'injury_risk': nearby_injury_risk,
                'prediction_score': prediction_score,
                'risk_level': 'High' if prediction_score > 0.7 else 'Medium' if prediction_score > 0.4 else 'Low'
            })
        
        self.future_predictions = pd.DataFrame(predictions)
        self.future_predictions = self.future_predictions.sort_values('prediction_score', ascending=False)
        
        print(f"\nFuture Hotspot Predictions for {future_year} ({weather_scenario} weather):")
        print(self.future_predictions[['cluster_id', 'center_lat', 'center_lon', 'prediction_score', 'risk_level']].head(10))
        
        return self
    
    def create_prediction_map(self, output_file='future_hotspots_map.html'):
        """Create an interactive map showing predicted future hotspots"""
        print("Creating prediction map...")
        
        # Center map on Amsterdam
        center_lat = self.coord_data_clean['latitude'].mean()
        center_lon = self.coord_data_clean['longitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Color mapping for risk levels
        risk_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
        
        # Add predicted hotspots
        for _, pred in self.future_predictions.iterrows():
            color = risk_colors[pred['risk_level']]
            
            folium.CircleMarker(
                location=[pred['center_lat'], pred['center_lon']],
                radius=max(5, pred['prediction_score'] * 20),
                popup=f"""
                <b>Predicted Hotspot</b><br>
                Cluster ID: {pred['cluster_id']}<br>
                Risk Level: {pred['risk_level']}<br>
                Prediction Score: {pred['prediction_score']:.3f}<br>
                Historical Accidents: {pred['historical_accidents']}<br>
                Weather Risk: {pred['weather_risk']:.3f}<br>
                Injury Risk: {pred['injury_risk']:.3f}
                """,
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
        
        # Add current accident locations as heat map
        heat_data = [[row['latitude'], row['longitude']] for _, row in 
                    self.coord_data_clean.sample(min(1000, len(self.coord_data_clean))).iterrows()]
        
        HeatMap(heat_data, radius=10, blur=15, max_zoom=1, name='Historical Accidents').add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        m.save(output_file)
        print(f"Map saved as {output_file}")
        
        return self
    
    def generate_report(self):
        """Generate a comprehensive prediction report"""
        print("Generating prediction report...")
        
        report = f"""
        BIKE ACCIDENT HOTSPOT PREDICTION REPORT
        =====================================
        
        ANALYSIS SUMMARY:
        - Total accidents analyzed: {len(self.coord_data_clean):,}
        - Hotspot clusters identified: {len(self.hotspot_clusters)}
        - Weather years analyzed: {len(self.df_weather)}
        - Injury locations analyzed: {len(self.df_injury_weekday):,}
        
        TOP 5 PREDICTED FUTURE HOTSPOTS:
        """
        
        for i, (_, pred) in enumerate(self.future_predictions.head(5).iterrows(), 1):
            report += f"""
        {i}. Cluster {pred['cluster_id']} - {pred['risk_level']} Risk
           Location: ({pred['center_lat']:.4f}, {pred['center_lon']:.4f})
           Prediction Score: {pred['prediction_score']:.3f}
           Historical Accidents: {pred['historical_accidents']}
           Weather Risk Factor: {pred['weather_risk']:.3f}
           Injury Risk Factor: {pred['injury_risk']:.3f}
        """
        
        report += f"""
        
        WEATHER ANALYSIS:
        - Latest year analyzed: {self.weather_analysis['accident_year'].max():.0f}
        - Average temperature: {self.weather_analysis['temp_C'].mean():.1f}°C
        - Average precipitation: {self.weather_analysis['precip_mm'].mean():.1f}mm
        - High-risk weather years: {self.weather_analysis['high_risk_weather'].sum()}
        
        INJURY SEVERITY ANALYSIS:
        - High-risk injury locations: {self.injury_analysis['high_risk_location'].sum()}
        - Average injury rate: {self.injury_analysis['injury_rate'].mean():.3f}
        - Locations with fatalities: {(self.injury_analysis['fatal_rate'] > 0).sum()}
        
        RECOMMENDATIONS:
        1. Focus safety interventions on high-risk predicted hotspots
        2. Increase monitoring during adverse weather conditions
        3. Implement targeted safety measures at locations with high injury severity
        4. Consider seasonal and weekday patterns for resource allocation
        """
        
        print(report)
        
        # Save report to file
        with open('hotspot_prediction_report.txt', 'w') as f:
            f.write(report)
        
        return report

def main():
    """Main execution function"""
    print("Starting Bike Accident Hotspot Prediction Analysis")
    print("=" * 50)
    
    # Initialize predictor
    predictor = BikeAccidentHotspotPredictor()
    
    # Run complete analysis pipeline
    try:
        predictor.load_data()
        predictor.preprocess_coordinate_data()
        predictor.apply_hdbscan_clustering()
        predictor.analyze_hotspot_characteristics()
        predictor.incorporate_weather_patterns()
        predictor.incorporate_injury_weekday_patterns()
        
        # Generate predictions for different scenarios
        print("\n" + "="*50)
        print("GENERATING PREDICTIONS")
        print("="*50)
        
        # Normal weather scenario
        predictor.predict_future_hotspots(future_year=2025, weather_scenario='normal')
        
        # Create visualization
        predictor.create_prediction_map('future_hotspots_normal_2025.html')
        
        # Generate harsh weather predictions
        print("\n" + "-"*30)
        predictor.predict_future_hotspots(future_year=2025, weather_scenario='harsh')
        predictor.create_prediction_map('future_hotspots_harsh_2025.html')
        
        # Generate final report
        print("\n" + "="*50)
        print("FINAL REPORT")
        print("="*50)
        predictor.generate_report()
        
        print("\nAnalysis completed successfully!")
        print("Generated files:")
        print("- future_hotspots_normal_2025.html")
        print("- future_hotspots_harsh_2025.html") 
        print("- hotspot_prediction_report.txt")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 