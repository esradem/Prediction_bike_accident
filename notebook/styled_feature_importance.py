# Styled Feature Importance Plotting Functions
# Custom color theme matching presentation

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set custom color theme
custom_colors = {
    'primary_blue': '#A8CEF1',    # Light blue from the image
    'secondary_blue': '#4B9FE1',  # Darker blue for emphasis
    'background': '#F5F9FF',      # Very light blue background
    'grid': '#E5E5E5',           # Light gray for grid
    'text': '#2F3545'            # Dark navy for text
}

def plot_feature_importance(model, model_name, X_columns):
    """
    Create styled feature importance plot for a given model
    
    Parameters:
    model: trained model with feature_importances_ attribute
    model_name: string name for the model (e.g., "Random Forest", "XGBoost")
    X_columns: column names for features
    """
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create figure with custom styling
    plt.figure(figsize=(10, 6), facecolor=custom_colors['background'])
    ax = plt.gca()
    ax.set_facecolor(custom_colors['background'])
    
    # Create bar plot with custom colors
    bars = sns.barplot(
        x='importance', 
        y='feature', 
        data=feature_importance,
        color=custom_colors['secondary_blue'],
        alpha=0.7
    )
    
    # Customize the plot
    plt.title(f'{model_name} - Feature Importance in Risk Prediction', 
              color=custom_colors['text'],
              pad=20,
              fontsize=14,
              fontweight='bold')
    
    plt.xlabel('Importance Score', 
              color=custom_colors['text'],
              fontsize=12)
    plt.ylabel('Features', 
              color=custom_colors['text'],
              fontsize=12)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(custom_colors['text'])
    ax.spines['bottom'].set_color(custom_colors['text'])
    ax.tick_params(colors=custom_colors['text'])
    
    # Add subtle grid
    ax.grid(True, axis='x', color=custom_colors['grid'], linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Add value labels on the bars
    for i, v in enumerate(feature_importance['importance']):
        ax.text(v + 0.01, i, f'{v:.3f}',
                color=custom_colors['text'],
                va='center',
                fontsize=10,
                fontweight='light')
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_comparison_feature_importance(rf_model, xgb_model, X_columns):
    """
    Create comparison plot of feature importance between Random Forest and XGBoost
    
    Parameters:
    rf_model: trained Random Forest model
    xgb_model: trained XGBoost model
    X_columns: column names for features
    """
    # Get feature importance for both models
    rf_importance = pd.DataFrame({
        'feature': X_columns,
        'importance': rf_model.feature_importances_,
        'model': 'Random Forest'
    })
    
    xgb_importance = pd.DataFrame({
        'feature': X_columns,
        'importance': xgb_model.feature_importances_,
        'model': 'XGBoost'
    })
    
    # Combine data
    combined_importance = pd.concat([rf_importance, xgb_importance])
    
    # Create figure
    plt.figure(figsize=(12, 6), facecolor=custom_colors['background'])
    ax = plt.gca()
    ax.set_facecolor(custom_colors['background'])
    
    # Create grouped bar plot
    sns.barplot(
        data=combined_importance,
        x='feature',
        y='importance',
        hue='model',
        palette=[custom_colors['secondary_blue'], custom_colors['primary_blue']]
    )
    
    # Customize the plot
    plt.title('Feature Importance Comparison: Random Forest vs XGBoost', 
              color=custom_colors['text'],
              pad=20,
              fontsize=14,
              fontweight='bold')
    
    plt.xlabel('Features', 
              color=custom_colors['text'],
              fontsize=12)
    plt.ylabel('Importance Score', 
              color=custom_colors['text'],
              fontsize=12)
    
    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(custom_colors['text'])
    ax.spines['bottom'].set_color(custom_colors['text'])
    ax.tick_params(colors=custom_colors['text'])
    
    # Style the legend
    legend = ax.legend()
    legend.get_frame().set_facecolor(custom_colors['background'])
    legend.get_frame().set_edgecolor(custom_colors['text'])
    for text in legend.get_texts():
        text.set_color(custom_colors['text'])
    
    # Add subtle grid
    ax.grid(True, axis='y', color=custom_colors['grid'], linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()

def train_and_plot_models(df_ams_cc):
    """
    Train both Random Forest and XGBoost models and create styled plots
    
    Parameters:
    df_ams_cc: DataFrame with accident data
    """
    # Prepare features for prediction
    X = df_ams_cc[['latitude', 'longitude', 'accident_year']]
    
    # Create target variable (1 for hotspot clusters, 0 for noise points)
    y = (df_ams_cc['cluster'] != -1).astype(int)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Train XGBoost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Make predictions for both models
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    
    # Print model performance
    print("Random Forest Model Performance:")
    print(classification_report(y_test, rf_pred))
    print("\nXGBoost Model Performance:")
    print(classification_report(y_test, xgb_pred))
    
    # Plot individual feature importance
    plot_feature_importance(rf_model, "Random Forest", X.columns)
    plot_feature_importance(xgb_model, "XGBoost", X.columns)
    
    # Plot comparison
    plot_comparison_feature_importance(rf_model, xgb_model, X.columns)
    
    return rf_model, xgb_model

def style_temporal_plot(df_ams_cc):
    """
    Create styled temporal trend plot
    
    Parameters:
    df_ams_cc: DataFrame with accident data
    """
    # Set matplotlib parameters for consistent styling
    plt.rcParams['figure.facecolor'] = custom_colors['background']
    plt.rcParams['axes.facecolor'] = custom_colors['background']
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = custom_colors['grid']
    plt.rcParams['grid.linestyle'] = '-'
    plt.rcParams['grid.alpha'] = 0.5

    # Create the figure
    plt.figure(figsize=(12, 6))

    # Convert year and create yearly accidents count
    df_ams_cc['year'] = df_ams_cc['accident_year'].astype(int)
    yearly_accidents = df_ams_cc['year'].value_counts().sort_index()

    # Create the plot with custom styling
    ax = plt.gca()

    # Plot the line with custom colors and styling
    plt.plot(yearly_accidents.index, yearly_accidents.values, 
             marker='o', 
             color=custom_colors['secondary_blue'],
             linewidth=3,
             markersize=8,
             markerfacecolor=custom_colors['primary_blue'],
             markeredgecolor=custom_colors['secondary_blue'],
             markeredgewidth=2)

    # Style the title and labels
    plt.title('Yearly Trend of Bike Accidents in Amsterdam', 
              color=custom_colors['text'],
              pad=20,
              fontsize=14,
              fontweight='bold')
    plt.xlabel('Year', color=custom_colors['text'], fontsize=12)
    plt.ylabel('Number of Accidents', color=custom_colors['text'], fontsize=12)

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(custom_colors['text'])
    ax.spines['bottom'].set_color(custom_colors['text'])
    ax.tick_params(colors=custom_colors['text'])

    # Adjust layout and display
    plt.tight_layout()
    plt.show() 