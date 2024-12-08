# CropInsight: Agricultural Data Analysis Platform 🌾

## Overview
CropInsight is a comprehensive web application designed to enhance agricultural productivity through data-driven insights. The platform analyzes multiple agricultural datasets to provide valuable insights into crop yields, environmental factors, and agricultural practices. It is tailored for agricultural professionals, data scientists, policymakers, and researchers.

## Project Links
- **GitHub Repository**: [https://github.com/ShangHuanChiang/CMSE-830](https://github.com/ShangHuanChiang/CMSE-830)
- **Data Source**: [https://www.kaggle.com/code/patelris/crop-yield-eda-viz/input](https://www.kaggle.com/code/patelris/crop-yield-eda-viz/input)
- **Documentation**: [final_project_streamlit.py](https://github.com/ShangHuanChiang/CMSE-830/blob/main/final_project/final_project_streamlit.py)
- **Streamlit link**:https://cmse-830-5pdmdsgmsvrxysf9vwfmyd.streamlit.app/

## Data Sources
The datasets used in this project are sourced from Kaggle's "Crop Yield Prediction" dataset, which includes:

- **Global Food & Agriculture Statistics**: Contains comprehensive agricultural data from FAO
- **Time period**: 1990-2013
- **Geographic coverage**: Global, country-level data
- **Data frequency**: Annual measurements
- **Source reliability**: Data validated by FAO (Food and Agriculture Organization)

## Features

### 📊 Interactive Data Visualization
- **Custom Scatter Plots**: Interactive visualization tool for exploring relationships between variables such as yield vs. rainfall or temperature
- **Correlation Heatmaps**: Dynamic heatmaps showing correlations between different agricultural variables for each crop type
- **Geographic Visualizations**: Interactive world map showing crop yields by region with temporal analysis
- **PCA Analysis**: Dimensionality reduction visualization to identify patterns in agricultural data
- **ARIMA Forecasting**: Time-series forecasting tool for predicting future crop yields
- **RBF-NN Interpolation**: Advanced interpolation using radial basis functions for estimating missing data points

### ⚙️ Technical Analysis Tools
- **Dataset Visualization**: Comparative analysis tools for multiple agricultural datasets
- **Missing Values Analysis**: Comprehensive missing data detection and visualization
- **MICE Implementation**: Advanced imputation technique for handling missing data

## Implementation Details

### Visualization Technologies
- **Plotly**: Used for interactive charts and geographic visualizations
- **Seaborn**: Implements statistical visualizations like correlation heatmaps
- **Matplotlib**: Supports basic plotting and customization

### Analysis Methods
1. **Statistical Analysis**
   - Correlation analysis between different agricultural variables
   - Time series analysis for yield predictions
   - Principal Component Analysis for pattern detection

2. **Machine Learning Models**
   - ARIMA models for time series forecasting
   - RBF Neural Networks for data interpolation
   - Linear Regression for yield predictions

3. **Data Processing**
   - MICE (Multiple Imputation by Chained Equations) for handling missing values
   - Data standardization and normalization
   - Temporal and spatial data aggregation

## Datasets Details

### 1. Agricultural Dataset (Integrated)
- **Content**: Comprehensive agricultural data including yields, environmental factors
- **Variables**: Crops, yields, pesticide usage, weather data
- **Time Range**: 1990-2013
- **Usage**: Primary dataset for analysis and visualization

### 2. Yield Dataset
- **Content**: Historical crop yield data
- **Variables**: Crop types, annual yields, regional data
- **Units**: Tonnes per hectare
- **Purpose**: Yield analysis and forecasting

### 3. Pesticides Dataset
- **Content**: Pesticide application data
- **Measurements**: Application rates, types, regional patterns
- **Units**: Tonnes of active ingredients
- **Purpose**: Agricultural input analysis

### 4. Temperature Dataset
- **Content**: Historical temperature records
- **Variables**: Average temperatures, regional variations
- **Units**: Degrees Celsius
- **Purpose**: Climate impact analysis

### 5. Rainfall Dataset
- **Content**: Precipitation records
- **Variables**: Annual rainfall, seasonal patterns
- **Units**: Millimeters per year
- **Purpose**: Weather impact analysis

## Streamlit Implementation
The application is built using Streamlit and includes:

1. **Interactive Dashboard**
   - Multi-page layout with sidebar navigation
   - Dynamic data loading and caching
   - Responsive design with custom styling

2. **Analysis Features**
   - Real-time data visualization
   - Interactive parameter selection
   - Dynamic filtering and aggregation
   - Custom color themes and styling

3. **Technical Components**
   - Data preprocessing and cleaning
   - Advanced statistical analysis
   - Machine learning model integration
   - Error handling and logging
