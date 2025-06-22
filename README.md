# Predicting Future Bike Accident Hotspots in Amsterdam 

### Using HDBSCAN Clustering & Multi-Source Data Analysis

A data-driven approach to predicting future bike accident hotspots using HDBSCAN clustering and multi-source data analysis.

## Project Overview

This project analyzes 29,763 bike accidents in Amsterdam (2014-2023) to identify spatial hotspots and predict future accident-prone areas. By combining coordinate data, weather patterns, and injury severity information, the system generates actionable insights for urban safety planning.

##  Project Goals

-  Identify high-risk intersections and accident clusters  
-  Analyze how weather and time affect bike accidents  
-  Predict future hotspots using machine learning  
-  Understand the relationship between accident severity and weather conditions

---

## Key Features

- **Spatial Clustering**: HDBSCAN algorithm identifies 170 distinct accident hotspots
- **Weather Integration**: Incorporates 7 years of meteorological data for risk assessment
- **Injury Analysis**: Evaluates severity patterns and fatal accident locations
- **Future Predictions**: Generates 2025 hotspot forecasts under different weather scenarios
- **Interactive Visualizations**: HTML maps with clickable hotspots and risk level indicators

## Dataset Information

| Dataset          | Description                                                                |
| ---------------- | -------------------------------------------------------------------------- |
| `bike_data.csv`  | Raw bike accident records (2014–2023) including geolocation and timestamps |
| `df_ams_cc.csv`  | Cleaned and clustered accident data with enriched features                 |
| `df_wea.csv`     | Hourly weather data (temperature, wind, precipitation, visibility)         |
| `gdf_joined.csv` | GeoDataFrame version of cleaned data for spatial plotting                  |
| `yearly_wea.csv` | Aggregated yearly weather features used in prediction models               |


## Methodology

### 1. Data Preprocessing
- Coordinate cleaning and Amsterdam area filtering
- Missing value handling and outlier detection
- Feature engineering for temporal patterns

### 2. HDBSCAN Clustering
- Minimum cluster size: 50 accidents
- Silhouette score: 0.486 (good clustering quality)
- Mapped clusters using Folium
- Noise point identification: 9,986 isolated incidents

### 3. Risk Assessment Model
The prediction model combines four components:
- **Base Risk (40%)**: Historical accident frequency (log-transformed)
- **Weather Risk (30%)**: Temperature, precipitation, wind, visibility factors
- **Temporal Weight (20%)**: Recent year emphasis (2015+ weighted)
- **Injury Risk (10%)**: Proximity to high-severity locations

- Trained Random Forest & XGBoost classifiers  
- Predicts whether a point is a future accident hotspot  
- Factors: location, time, weather, injury severity

### 4. Future Prediction
- Risk level classification: High (>0.7), Medium (0.4-0.7), Low (<0.4)
- Weather scenario modeling: Normal vs Harsh conditions
- 2025 hotspot forecasts with confidence intervals

## Key Results

### Clustering Performance
- **170 hotspot clusters** identified across Amsterdam
- **Largest hotspot**: 1,189 accidents (Ringweg-West area)
- **Average accidents per hotspot**: 116.3
- **Top accident-prone street**: Ringweg-West (261 accidents)

### Temporal Trends
- **95% increase** in accidents from 2017 (2,384) to 2023 (4,660)
- COVID-19 impact visible in 2020 data (-22.9% decrease)
- Consistent growth pattern in post-pandemic years

### Injury Severity Analysis
- **54.4%** average injury rate across all accidents
- **36 fatal accidents** identified in dataset
- **34 high-risk injury locations** requiring immediate attention

### 2025 Predictions
- **94 medium-risk** and **76 low-risk** hotspot predictions
- **Highest prediction score**: 0.555 (Cluster 117 - Ringweg-West)
- **Weather sensitivity**: 6.7% average risk increase under harsh conditions

## Installation and Usage

### Prerequisites
```bash
Python 3.8+
pandas
numpy
scikit-learn
hdbscan
folium
matplotlib
seaborn
```

### Installation
```bash
git clone https://github.com/esradem/bike-accident-hotspot-prediction.git
cd bike-accident-hotspot-prediction
pip install -r requirements.txt
```

### Running the Analysis
```bash
# Main prediction script
python predict_future_hotspots.py

# Generate additional visualizations
python visualize_results.py
```

## Generated Outputs

### Interactive Maps
- `future_hotspots_normal_2025.html`: Normal weather scenario predictions
- `future_hotspots_harsh_2025.html`: Harsh weather scenario predictions

### Analysis Reports
- `hotspot_prediction_report.txt`: Comprehensive analysis summary
- `hotspot_analysis_comprehensive.png`: Statistical visualizations
- `detailed_hotspot_analysis.png`: Detailed hotspot characteristics

### Features of Interactive Maps
- Clickable hotspot markers with detailed information
- Risk level color coding (Red/Orange/Yellow)
- Historical accident heat map overlay
- Layer control for different visualization modes
- Popup statistics for each hotspot cluster

<<<<<<< HEAD
## Project Structure

```
bike-accident-hotspot-prediction/
├── data/
│   └── cleaned/
│       ├── df_ams_cc.csv
│       ├── yearly_wea.csv
│       └── gdf_joined.csv
├── predict_future_hotspots.py
├── visualize_results.py
├── requirements.txt
├── README.md
└── presentation/
   
```

=======
>>>>>>> b11cbc412a16f2f6571a16c77fccad82c66d4dbb
## Technical Implementation

### BikeAccidentHotspotPredictor Class
The main analysis is implemented through a comprehensive class with the following methods:

- `load_data()`: Loads three independent datasets without merging
- `preprocess_coordinate_data()`: Cleans and prepares spatial data
- `apply_hdbscan_clustering()`: Performs spatial clustering analysis
- `analyze_hotspot_characteristics()`: Extracts cluster statistics
- `incorporate_weather_patterns()`: Integrates meteorological data
- `incorporate_injury_weekday_patterns()`: Analyzes severity patterns
- `predict_future_hotspots()`: Generates 2025 predictions
- `create_prediction_map()`: Produces interactive visualizations
- `generate_report()`: Creates comprehensive analysis summary

### Key Algorithms
- **HDBSCAN**: Hierarchical density-based clustering for hotspot identification
- **Silhouette Analysis**: Clustering quality validation
- **Feature Scaling**: StandardScaler and MinMaxScaler for data normalization
- **Risk Scoring**: Weighted combination of historical, weather, temporal, and injury factors

## Applications and Impact

### Urban Planning
- Infrastructure improvement prioritization
- Traffic safety intervention planning
- Resource allocation for accident prevention

### Public Safety
- Weather-based safety alert systems
- Targeted safety campaign locations
- Emergency response optimization

### Policy Making
- Evidence-based cycling infrastructure decisions
- Data-driven urban mobility planning
- Accident prevention strategy development

## Future Work

### Immediate Enhancements
- Real-time traffic data integration
- Seasonal and monthly prediction models
- Mobile application for cyclist route optimization
- Integration with bike-sharing system data

### Research Extensions
- Machine learning models for dynamic risk scoring
- Expansion to other Dutch cities
- Pedestrian accident analysis integration
- Social and economic factor incorporation

### Technical Improvements
- API development for real-time predictions
- Dashboard creation for city planners
- Automated alert system implementation
- Performance optimization for larger datasets

## Validation and Limitations

### Model Validation
- Cross-validation with historical patterns
- Silhouette score analysis for clustering quality
- Weather scenario sensitivity testing
- Geographic boundary validation

### Current Limitations
- Limited to Amsterdam metropolitan area
- Weather data aggregated at yearly level
- No real-time traffic condition integration
- Prediction horizon limited to one year

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## Acknowledgments

- Amsterdam city data providers for accident and geographic information
- Weather data sources for meteorological information
- HDBSCAN algorithm developers for clustering methodology
- Folium library for interactive mapping capabilities

## Contact

For questions, suggestions, or collaboration opportunities, please open an issue in this repository.

## Citation

If you use this work in your research, please cite:

```
Bike Accident Hotspot Prediction in Amsterdam: A Data-Driven Approach Using HDBSCAN Clustering
[Esra Demirel], [2025]
GitHub: https://github.com/esradem/bike-accident-hotspot-prediction
```
