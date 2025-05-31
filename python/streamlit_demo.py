#!/usr/bin/env python3
"""
Streamlit Demo Application for Bike Accident Hotspot Prediction
This app provides an interactive interface to explore all analysis results
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, MarkerCluster
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

# Import the prediction class
try:
    from predict_future_hotspots import BikeAccidentHotspotPredictor
except ImportError:
    st.error("Could not import BikeAccidentHotspotPredictor. Please ensure the module is available.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="🚴‍♂️ Bike Accident Hotspot Prediction Demo",
    page_icon="🚴‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #F24236;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def load_and_run_analysis():
    """Load data and run the complete analysis pipeline"""
    with st.spinner("🔄 Loading data and running analysis..."):
        try:
            predictor = BikeAccidentHotspotPredictor()
            predictor.load_data()
            predictor.preprocess_coordinate_data()
            predictor.apply_hdbscan_clustering()
            predictor.analyze_hotspot_characteristics()
            predictor.incorporate_weather_patterns()
            predictor.incorporate_injury_weekday_patterns()
            predictor.predict_future_hotspots(future_year=2025, weather_scenario='normal')
            
            st.session_state.predictor = predictor
            st.session_state.analysis_complete = True
            return predictor
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")
            return None

def create_overview_metrics(predictor):
    """Create overview metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Total Accidents",
            value=f"{len(predictor.coord_data_clean):,}",
            help="Total number of accidents analyzed"
        )
    
    with col2:
        st.metric(
            label="🔍 Hotspot Clusters",
            value=len(predictor.hotspot_clusters),
            help="Number of accident hotspot clusters identified"
        )
    
    with col3:
        high_risk = (predictor.future_predictions['risk_level'] == 'High').sum()
        st.metric(
            label="⚠️ High Risk Predictions",
            value=high_risk,
            help="Number of high-risk hotspots predicted for 2025"
        )
    
    with col4:
        avg_score = predictor.future_predictions['prediction_score'].mean()
        st.metric(
            label="📊 Avg Prediction Score",
            value=f"{avg_score:.3f}",
            help="Average prediction score across all hotspots"
        )

def create_interactive_map(predictor, map_type="hotspots"):
    """Create interactive maps using Folium"""
    # Center map on Amsterdam
    center_lat = predictor.coord_data_clean['latitude'].mean()
    center_lon = predictor.coord_data_clean['longitude'].mean()
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    if map_type == "hotspots":
        # Add predicted hotspots
        risk_colors = {'High': 'red', 'Medium': 'orange', 'Low': 'yellow'}
        
        for _, pred in predictor.future_predictions.head(20).iterrows():
            color = risk_colors.get(pred['risk_level'], 'blue')
            
            folium.CircleMarker(
                location=[pred['center_lat'], pred['center_lon']],
                radius=max(5, pred['prediction_score'] * 20),
                popup=f"""
                <b>Predicted Hotspot</b><br>
                Cluster ID: {pred['cluster_id']}<br>
                Risk Level: {pred['risk_level']}<br>
                Prediction Score: {pred['prediction_score']:.3f}<br>
                Historical Accidents: {pred['historical_accidents']}
                """,
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    elif map_type == "heatmap":
        # Create heat map of all accidents
        heat_data = [[row['latitude'], row['longitude']] for _, row in 
                    predictor.coord_data_clean.sample(min(2000, len(predictor.coord_data_clean))).iterrows()]
        
        HeatMap(heat_data, radius=15, blur=20, max_zoom=1).add_to(m)
    
    return m

def create_plotly_charts(predictor):
    """Create interactive Plotly charts"""
    
    # 1. Hotspot Distribution
    fig_dist = px.histogram(
        predictor.hotspot_analysis, 
        x='accident_count',
        nbins=30,
        title="Distribution of Accident Counts Across Hotspots",
        labels={'accident_count': 'Number of Accidents per Hotspot', 'count': 'Frequency'},
        color_discrete_sequence=['#2E86AB']
    )
    fig_dist.update_layout(showlegend=False)
    
    # 2. Top Hotspots
    top_10 = predictor.hotspot_analysis.head(10)
    fig_top = px.bar(
        top_10,
        x='accident_count',
        y=[f"Cluster {id}" for id in top_10['cluster_id']],
        orientation='h',
        title="Top 10 Hotspots by Accident Count",
        labels={'accident_count': 'Number of Accidents', 'y': 'Cluster ID'},
        color='accident_count',
        color_continuous_scale='Reds'
    )
    
    # 3. Weather Trends
    weather_data = predictor.weather_analysis
    fig_weather = px.line(
        weather_data,
        x='accident_year',
        y='total_accidents',
        title="Accident Trends Over Years",
        labels={'accident_year': 'Year', 'total_accidents': 'Total Accidents'},
        markers=True
    )
    fig_weather.update_traces(line_color='#2F9B69', marker_color='#2F9B69')
    
    # 4. Risk Level Distribution
    risk_counts = predictor.future_predictions['risk_level'].value_counts()
    fig_risk = px.pie(
        values=risk_counts.values,
        names=risk_counts.index,
        title="Distribution of Risk Levels for Future Hotspots",
        color_discrete_map={'High': '#F24236', 'Medium': '#F6AE2D', 'Low': '#2F9B69'}
    )
    
    # 5. Prediction Components
    components = ['base_risk', 'weather_risk', 'temporal_weight', 'injury_risk']
    avg_components = [predictor.future_predictions[comp].mean() for comp in components]
    
    fig_components = px.bar(
        x=['Base Risk', 'Weather Risk', 'Temporal Weight', 'Injury Risk'],
        y=avg_components,
        title="Average Prediction Score Components",
        labels={'x': 'Component', 'y': 'Average Score'},
        color=avg_components,
        color_continuous_scale='Viridis'
    )
    
    return fig_dist, fig_top, fig_weather, fig_risk, fig_components

def create_weather_analysis_charts(predictor):
    """Create weather analysis charts"""
    weather_data = predictor.weather_analysis
    
    # Weather factors subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Temperature Trend', 'Precipitation Trend', 'Wind Speed Trend', 'Weather Risk Years'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    # Temperature
    fig.add_trace(
        go.Scatter(x=weather_data['accident_year'], y=weather_data['temp_C'],
                  mode='lines+markers', name='Temperature', line_color='#F24236'),
        row=1, col=1
    )
    
    # Precipitation
    fig.add_trace(
        go.Scatter(x=weather_data['accident_year'], y=weather_data['precip_mm'],
                  mode='lines+markers', name='Precipitation', line_color='#2E86AB'),
        row=1, col=2
    )
    
    # Wind Speed
    fig.add_trace(
        go.Scatter(x=weather_data['accident_year'], y=weather_data['wind_avg_ms'],
                  mode='lines+markers', name='Wind Speed', line_color='#2F9B69'),
        row=2, col=1
    )
    
    # High-risk weather years pie chart
    risk_counts = weather_data['high_risk_weather'].value_counts()
    fig.add_trace(
        go.Pie(labels=['Normal Weather', 'High-Risk Weather'], values=risk_counts.values,
               marker_colors=['lightblue', 'salmon']),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text="Weather Analysis Dashboard")
    return fig

def display_statistical_summary(predictor):
    """Display comprehensive statistical summary"""
    
    st.markdown("### 📊 Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Spatial Coverage:**
        - Total accidents analyzed: {len(predictor.coord_data_clean):,}
        - Geographic area: Amsterdam
        - Latitude range: {predictor.coord_data_clean['latitude'].min():.4f} - {predictor.coord_data_clean['latitude'].max():.4f}
        - Longitude range: {predictor.coord_data_clean['longitude'].min():.4f} - {predictor.coord_data_clean['longitude'].max():.4f}
        """)
    
    with col2:
        st.markdown(f"""
        **Temporal Coverage:**
        - Unique accident years: {predictor.coord_data_clean['accident_year'].nunique()}
        - Year range: {predictor.coord_data_clean['accident_year'].min():.0f} - {predictor.coord_data_clean['accident_year'].max():.0f}
        - Weather data years: {len(predictor.weather_analysis)}
        """)
    
    st.markdown("### 🎯 Clustering Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Clusters", len(predictor.hotspot_clusters))
    with col2:
        st.metric("Largest Hotspot", f"{predictor.hotspot_analysis.iloc[0]['accident_count']} accidents")
    with col3:
        st.metric("Avg Accidents/Hotspot", f"{predictor.hotspot_analysis['accident_count'].mean():.1f}")
    
    st.markdown("### 🌤️ Weather Analysis")
    weather_data = predictor.weather_analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Avg Temperature", f"{weather_data['temp_C'].mean():.1f}°C")
    with col2:
        st.metric("Avg Precipitation", f"{weather_data['precip_mm'].mean():.1f}mm")
    with col3:
        st.metric("Avg Wind Speed", f"{weather_data['wind_avg_ms'].mean():.1f}m/s")
    
    st.markdown("### 🚑 Injury Analysis")
    injury_data = predictor.injury_analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High-Risk Locations", injury_data['high_risk_location'].sum())
    with col2:
        st.metric("Avg Injury Rate", f"{injury_data['injury_rate'].mean():.3f}")
    with col3:
        st.metric("Locations with Fatalities", (injury_data['fatal_rate'] > 0).sum())

def display_prediction_results(predictor):
    """Display prediction results and analysis"""
    
    st.markdown("### 🔮 Future Predictions (2025)")
    
    # Prediction summary
    predictions = predictor.future_predictions
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("High Risk", (predictions['risk_level'] == 'High').sum())
    with col2:
        st.metric("Medium Risk", (predictions['risk_level'] == 'Medium').sum())
    with col3:
        st.metric("Low Risk", (predictions['risk_level'] == 'Low').sum())
    with col4:
        st.metric("Max Score", f"{predictions['prediction_score'].max():.3f}")
    
    # Top predictions table
    st.markdown("#### 🏆 Top 10 Predicted Hotspots")
    top_predictions = predictions.head(10)[['cluster_id', 'center_lat', 'center_lon', 
                                          'prediction_score', 'risk_level', 'historical_accidents']]
    st.dataframe(top_predictions, use_container_width=True)
    
    # Weather scenario comparison
    st.markdown("#### 🌦️ Weather Scenario Analysis")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        st.markdown("**Normal Weather Scenario:**")
        st.write("Current prediction results shown above")
    
    with scenario_col2:
        if st.button("🌩️ Run Harsh Weather Analysis"):
            with st.spinner("Analyzing harsh weather scenario..."):
                predictor.predict_future_hotspots(future_year=2025, weather_scenario='harsh')
                harsh_predictions = predictor.future_predictions.copy()
                
                st.markdown("**Harsh Weather Results:**")
                st.metric("High Risk Hotspots", (harsh_predictions['risk_level'] == 'High').sum())
                st.metric("Avg Prediction Score", f"{harsh_predictions['prediction_score'].mean():.3f}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚴‍♂️ Bike Accident Hotspot Prediction Demo</h1>', unsafe_allow_html=True)
    st.markdown("### Interactive Analysis of Amsterdam Bike Accident Data")
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📊 Analysis Dashboard", "🗺️ Interactive Maps", 
         "📈 Detailed Charts", "🌤️ Weather Analysis", "🔮 Predictions", 
         "📋 Statistical Summary", "💾 Data Export"]
    )
    
    # Load data button
    if not st.session_state.analysis_complete:
        st.sidebar.markdown("---")
        if st.sidebar.button("🚀 Load Data & Run Analysis", type="primary"):
            predictor = load_and_run_analysis()
            if predictor:
                st.sidebar.success("✅ Analysis completed!")
        else:
            st.warning("⚠️ Please click 'Load Data & Run Analysis' to begin.")
            st.stop()
    
    predictor = st.session_state.predictor
    
    # Page routing
    if page == "🏠 Overview":
        st.markdown("## 📋 Project Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            This demo showcases a comprehensive bike accident hotspot prediction system for Amsterdam. 
            The analysis combines multiple data sources to identify current accident hotspots and predict 
            future high-risk areas.
            
            **Key Features:**
            - 🎯 **HDBSCAN Clustering** for hotspot identification
            - 🌤️ **Weather Pattern Analysis** for risk assessment
            - 🚑 **Injury Severity Integration** for comprehensive risk scoring
            - 🔮 **Future Predictions** with multiple weather scenarios
            - 🗺️ **Interactive Visualizations** for exploration
            """)
        
        with col2:
            st.markdown("""
            **Data Sources:**
            - Accident coordinates
            - Weather data
            - Injury severity records
            - Temporal patterns
            """)
        
        # Overview metrics
        st.markdown("## 📊 Key Metrics")
        create_overview_metrics(predictor)
        
        # Quick insights
        st.markdown("## 💡 Quick Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="success-box">
            <h4>🎯 Hotspot Detection</h4>
            Successfully identified distinct accident concentration areas using advanced clustering techniques.
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Risk Factors</h4>
            Weather conditions and injury patterns significantly influence accident risk prediction.
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>🔮 Future Predictions</h4>
            Generated risk scores for 2025 based on historical patterns and environmental factors.
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "📊 Analysis Dashboard":
        st.markdown("## 📊 Analysis Dashboard")
        
        # Create charts
        fig_dist, fig_top, fig_weather, fig_risk, fig_components = create_plotly_charts(predictor)
        
        # Display charts in tabs
        tab1, tab2, tab3 = st.tabs(["📈 Distribution Analysis", "🏆 Top Hotspots", "🌤️ Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_dist, use_container_width=True)
            with col2:
                st.plotly_chart(fig_risk, use_container_width=True)
        
        with tab2:
            st.plotly_chart(fig_top, use_container_width=True)
            st.plotly_chart(fig_components, use_container_width=True)
        
        with tab3:
            st.plotly_chart(fig_weather, use_container_width=True)
    
    elif page == "🗺️ Interactive Maps":
        st.markdown("## 🗺️ Interactive Maps")
        
        map_type = st.selectbox(
            "Choose map type:",
            ["hotspots", "heatmap"],
            format_func=lambda x: "🎯 Predicted Hotspots" if x == "hotspots" else "🔥 Accident Heatmap"
        )
        
        # Create and display map
        m = create_interactive_map(predictor, map_type)
        st_folium(m, width=700, height=500)
        
        if map_type == "hotspots":
            st.markdown("""
            **Map Legend:**
            - 🔴 Red: High Risk
            - 🟠 Orange: Medium Risk  
            - 🟡 Yellow: Low Risk
            - Size indicates prediction score
            """)
        else:
            st.markdown("**Heatmap shows accident density across Amsterdam**")
    
    elif page == "📈 Detailed Charts":
        st.markdown("## 📈 Detailed Analysis Charts")
        
        # Hotspot characteristics
        st.markdown("### 🎯 Hotspot Characteristics")
        
        # Scatter plot: Historical vs Predicted
        top_hotspots = predictor.hotspot_analysis.head(10)
        top_predictions = predictor.future_predictions.head(10)
        
        fig_scatter = px.scatter(
            x=top_hotspots['accident_count'],
            y=top_predictions['prediction_score'],
            title="Historical vs Predicted Risk",
            labels={'x': 'Historical Accident Count', 'y': 'Future Prediction Score'},
            size=top_predictions['prediction_score'],
            color=top_predictions['risk_level'],
            color_discrete_map={'High': '#F24236', 'Medium': '#F6AE2D', 'Low': '#2F9B69'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Geographic distribution
        amsterdam_hotspots = top_hotspots[
            (top_hotspots['center_lat'].between(52.31, 52.42)) &
            (top_hotspots['center_lon'].between(4.82, 4.98))
        ]
        
        if len(amsterdam_hotspots) > 0:
            fig_geo = px.scatter_mapbox(
                amsterdam_hotspots,
                lat='center_lat',
                lon='center_lon',
                size='accident_count',
                color='accident_count',
                hover_data=['cluster_id'],
                mapbox_style="open-street-map",
                title="Geographic Distribution of Top Hotspots",
                zoom=11
            )
            st.plotly_chart(fig_geo, use_container_width=True)
    
    elif page == "🌤️ Weather Analysis":
        st.markdown("## 🌤️ Weather Analysis")
        
        # Weather dashboard
        fig_weather_dash = create_weather_analysis_charts(predictor)
        st.plotly_chart(fig_weather_dash, use_container_width=True)
        
        # Weather statistics
        weather_data = predictor.weather_analysis
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Weather Statistics")
            st.dataframe(weather_data.describe(), use_container_width=True)
        
        with col2:
            st.markdown("### ⚠️ High-Risk Weather Analysis")
            high_risk_years = weather_data[weather_data['high_risk_weather'] == 1]
            if len(high_risk_years) > 0:
                st.dataframe(high_risk_years[['accident_year', 'temp_C', 'precip_mm', 'wind_avg_ms']], 
                           use_container_width=True)
            else:
                st.info("No high-risk weather years identified in the dataset.")
    
    elif page == "🔮 Predictions":
        st.markdown("## 🔮 Future Predictions")
        display_prediction_results(predictor)
    
    elif page == "📋 Statistical Summary":
        st.markdown("## 📋 Statistical Summary")
        display_statistical_summary(predictor)
        
        # Additional detailed tables
        st.markdown("### 📊 Detailed Data Tables")
        
        tab1, tab2, tab3 = st.tabs(["🎯 Hotspot Analysis", "🔮 Predictions", "🌤️ Weather Data"])
        
        with tab1:
            st.dataframe(predictor.hotspot_analysis, use_container_width=True)
        
        with tab2:
            st.dataframe(predictor.future_predictions, use_container_width=True)
        
        with tab3:
            st.dataframe(predictor.weather_analysis, use_container_width=True)
    
    elif page == "💾 Data Export":
        st.markdown("## 💾 Data Export")
        
        st.markdown("Download analysis results and visualizations:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export hotspot analysis
            hotspot_csv = predictor.hotspot_analysis.to_csv(index=False)
            st.download_button(
                label="📊 Download Hotspot Analysis",
                data=hotspot_csv,
                file_name="hotspot_analysis_results.csv",
                mime="text/csv"
            )
            
            # Export predictions
            predictions_csv = predictor.future_predictions.to_csv(index=False)
            st.download_button(
                label="🔮 Download Predictions",
                data=predictions_csv,
                file_name="future_predictions_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export weather analysis
            weather_csv = predictor.weather_analysis.to_csv(index=False)
            st.download_button(
                label="🌤️ Download Weather Analysis",
                data=weather_csv,
                file_name="weather_analysis_results.csv",
                mime="text/csv"
            )
            
            # Summary statistics
            summary_stats = {
                'total_accidents': len(predictor.coord_data_clean),
                'total_clusters': len(predictor.hotspot_clusters),
                'high_risk_predictions': (predictor.future_predictions['risk_level'] == 'High').sum(),
                'avg_prediction_score': predictor.future_predictions['prediction_score'].mean()
            }
            summary_df = pd.DataFrame([summary_stats])
            summary_csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📋 Download Summary Stats",
                data=summary_csv,
                file_name="analysis_summary_stats.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🚴‍♂️ Bike Accident Hotspot Prediction Demo | Built with Streamlit | 
    Data: Amsterdam Bike Accidents | Analysis: HDBSCAN Clustering + Weather Integration
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 