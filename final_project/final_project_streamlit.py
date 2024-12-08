import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
from PIL import Image
import requests
from io import BytesIO
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from scipy.interpolate import RBFInterpolator
import missingno as msno
import matplotlib.pyplot as plt
from functools import lru_cache
import logging
from sklearn.linear_model import LinearRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.ticker as mticker

logging.basicConfig(level=logging.ERROR)

THEME_COLORS = {
    'primary': '#26734D',
    'secondary': '#2C3E50',
    'background': '#f2f8f3',
    'accent': '#388E3C'
}

st.set_page_config(
    page_title="CropInsight",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)
def load_data():
    base_url = "https://raw.githubusercontent.com/ShangHuanChiang/CMSE-830/main/final_project/"
    datasets = {
        "agricultural": "agricultural_dataset.csv",
        "pesticides": "pesticides.csv",
        "temp": "temp.csv",
        "yield": "yield.csv",
        "rainfall": "rainfall.csv"
    }
    
    data = {}
    for key, file in datasets.items():
        df = pd.read_csv(f"{base_url}{file}")
        
        if key == 'yield':
            df = df.drop(columns=["Domain Code", "Domain", "Area Code", "Element Code", 
                                  "Item Code", "Element", "Year Code", "Unit"], errors='ignore')
            df.rename(columns={'Year': 'year', 'Item': 'Crops', 'Value': 'yield'}, inplace=True)
        else:
            df.rename(columns={'Year': 'year'}, inplace=True)
        
        # Optimize memory usage
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
            
        data[key] = df
    return data

@st.cache_data
def load_and_process_bg_image():
    try:
        response = requests.get(
            "https://img.freepik.com/free-photo/detail-rice-plant-sunset-valencia-with-plantation-out-focus-rice-grains-plant-seed_181624-25838.jpg",
            timeout=5
        )
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
            image = image.resize((1920, 1080), Image.Resampling.LANCZOS)
            buffered = BytesIO()
            image.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
        else:
            return None
    except Exception as e:
        logging.error(f"Failed to load background image: {e}")
        return None

def apply_styling(img_str):
    if img_str:
        background_style = f"""
        background-image: url("data:image/jpg;base64,{img_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        """
    else:
        background_style = "background-color: var(--background-color);"

    st.markdown(f"""
        <style>
        :root {{
            --primary-color: {THEME_COLORS['primary']};
            --secondary-color: {THEME_COLORS['secondary']};
            --background-color: {THEME_COLORS['background']};
            --accent-color: {THEME_COLORS['accent']};
        }}

        .stApp {{
            {background_style}
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(255,255,255,0.85);
            z-index: -1;
        }}

        .text-area {{
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        h1 {{
            color: var(--primary-color);
            font-size: 2.5rem;
            font-weight: 700;
            text-align: center;
            margin-bottom: 2rem;
        }}

        h2 {{
            color: var(--accent-color);
            font-size: 2rem;
            margin: 1.5rem 0;
            text-align: center;
        }}

        h3 {{
            color: var(--primary-color);
        }}

        .info-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 1rem;
        }}

        .info-item {{
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}

        .features-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }}

        .feature-section {{
            background: rgba(255,255,255,0.95);
            border-radius: 8px;
            padding: 1rem;
        }}

        .get-started-section {{
            text-align: center;
            margin-top: 2rem;
        }}
        </style>
    """, unsafe_allow_html=True)

def show_homepage():
    st.markdown("""
        <div class='centered-title'>
            <h1>🌾 CropInsight: Enhancing Agricultural Productivity</h1>
        </div>

        <div class='text-area'>
            <h2>Welcome to CropInsight</h2>
            <p style='text-align: center; font-size: 1.2rem; line-height: 1.6;'>
            CropInsight aims to enhance agricultural productivity by providing tools to analyze how various factors affect crop yields. 
            It is designed for agricultural professionals, data scientists, policymakers, and researchers to make data-driven decisions for improved agricultural outcomes.
            </p>
        </div>

        <div class='section-header'>
            <h2 style='margin-bottom: 2rem;'>Understanding CropInsight</h2>
        </div>

        <style>
        .grid-container {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            padding: 1rem;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .grid-item {
            padding: 1.5rem;
            border-radius: 8px;
            background: #f8f9fa;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .grid-item h3 {
            color: var(--primary-color);
            margin-bottom: 1rem;
            font-size: 1.5rem;
            text-align: center;
            border-bottom: 2px solid var(--accent-color);
            padding-bottom: 0.5rem;
        }
        .grid-item p {
            font-size: 1.1rem;
            line-height: 1.6;
            text-align: left;
            margin: 0;
            padding: 0.5rem;
        }
        </style>

        <div class='grid-container'>
            <div class='grid-item'>
                <h3>Why</h3>
                <p>To improve agricultural productivity by analyzing factors affecting crop yields.</p>
            </div>
            <div class='grid-item'>
                <h3>Who</h3>
                <p>Agricultural experts, data scientists, policymakers, and researchers.</p>
            </div>
            <div class='grid-item'>
                <h3>What</h3>
                <p>Tools for analyzing datasets related to crop yields, rainfall, temperature, and pesticide usage.</p>
            </div>
            <div class='grid-item'>
                <h3>When</h3>
                <p>Useful for planning agricultural cycles or researching climate impacts on yields.</p>
            </div>
            <div class='grid-item'>
                <h3>Where</h3>
                <p>Data sourced from various global agricultural datasets.</p>
            </div>
            <div class='grid-item'>
                <h3>How</h3>
                <p>Combining historical data analysis, advanced visualization, and predictive modeling.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
def create_customer_visualization(df):
    st.subheader("Custom Data Visualization")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.write("Not enough numeric columns for scatter plot.")
        return

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis", numeric_cols)
    with col2:
        y_axis = st.selectbox("Select Y-axis", numeric_cols)

    color_by_options = []
    if 'Crops' in df.columns:
        color_by_options.append('Crops')
    if 'Area' in df.columns:
        color_by_options.append('Area')

    color_by = st.selectbox("Color by", color_by_options) if color_by_options else None

    fig = px.scatter(df, x=x_axis, y=y_axis, color=color_by,
                     title=f"{y_axis} vs {x_axis}" + (f" by {color_by}" if color_by else ""))
    st.plotly_chart(fig, use_container_width=True)

def create_correlation_heatmap(df):
    st.subheader("Correlation Analysis")
    df = df.rename(columns={'Item': 'Crops'})
    
    selected_crop = st.selectbox("Select Crop", ['All'] + list(df['Crops'].unique()))
    crop_data = df[df['Crops'] == selected_crop] if selected_crop != 'All' else df

    numeric_cols = crop_data.select_dtypes(include=[np.number]).columns
    corr_matrix = crop_data[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale="viridis",
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=f"Correlation Heatmap for {selected_crop}",
        width=700,
        height=700,
        xaxis_title="",
        yaxis_title="",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_tickangle=45,
        yaxis_autorange='reversed'
    )
    st.plotly_chart(fig, use_container_width=True)
    
def add_chart_usage_instructions():
    st.markdown("""
        <div class='text-area'>
            <h4>Chart Usage Instructions:</h4>
            <ul>
                <li>Use the selection boxes above to filter the data</li>
                <li>Hover over points to see detailed information</li>
                <li>Click and drag to zoom</li>
                <li>Double click to reset the view</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)    

def create_geographic_time_series(df):
    st.subheader("Geographic Time Series")
    required_cols = ['Crops', 'Area', 'yield']
    if any(col not in df.columns for col in required_cols):
        st.write("The required columns (Crops, Area, yield) are not available.")
        return

    year = st.slider("Select Year",
                     min_value=int(df['year'].min()),
                     max_value=int(df['year'].max()))
    selected_crop = st.selectbox("Select Crop for Map", df['Crops'].unique())
    year_data = df[(df['year'] == year) & (df['Crops'] == selected_crop)]

    fig = px.scatter_geo(year_data, locations="Area", locationmode="country names",
                         color="yield", hover_name="Area", size="yield",
                         title=f"{selected_crop} Yield by Region ({year})")
    st.plotly_chart(fig, use_container_width=True)

def create_pca_analysis(df):
    st.subheader("PCA Analysis")
    df = df.rename(columns={'Item': 'Crops'})
    selected_crop = st.selectbox("Select Crop for PCA", ['All'] + list(df['Crops'].unique()))
    df_pca = df[df['Crops'] == selected_crop] if selected_crop != 'All' else df
    numeric_cols = df_pca.select_dtypes(include=[np.number]).columns
    X = StandardScaler().fit_transform(df_pca[numeric_cols])
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    fig = px.scatter(x=components[:, 0], y=components[:, 1],
                    color=df_pca['Area'],
                    title=f"PCA Analysis for {selected_crop}",
                    labels={
                        'x': f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)",
                        'y': f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)"
                    })
    st.plotly_chart(fig, use_container_width=True)
    add_chart_usage_instructions()

def create_arima_forecast(df):
    st.subheader("ARIMA Forecast")
    if 'Crops' not in df.columns or 'yield' not in df.columns:
        st.write("Missing 'Crops' or 'yield' columns for ARIMA forecasting.")
        return

    selected_crop = st.selectbox("Select Crop for Forecast", df['Crops'].unique())
    crop_data = df[df['Crops'] == selected_crop]['yield'].values
    if len(crop_data) < 10:
        st.write("Not enough data points for ARIMA forecasting.")
        return

    # Calculate test size as 20% of data
    test_size = int(len(crop_data) * 0.2)
    train_data = crop_data[:-test_size]
    test_data = crop_data[-test_size:]

    p_value = st.slider("Select AR(p) Parameter", 1, 5, 2)
    model = ARIMA(train_data, order=(p_value, 1, 1))
    results = model.fit()
    forecast = results.forecast(steps=test_size)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_data, name="Training Data", mode='lines',
                            line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=np.arange(len(train_data), len(crop_data)), 
                            y=test_data, name="Testing Data", mode='lines',
                            line=dict(color='green')))
    fig.add_trace(go.Scatter(x=np.arange(len(train_data), len(crop_data)), 
                            y=forecast, name="Forecast", mode='lines',
                            line=dict(color='red', dash='dash')))
    
    fig.update_layout(
        title=f"ARIMA Forecast for {selected_crop}",
        xaxis_title="Time",
        yaxis_title="Yield",
        showlegend=True,
        width=800,
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # Add model information
    st.markdown("""
        <div class='text-area'>
            <h4>Model Information:</h4>
            <ul>
                <li><strong>Training Data:</strong> First 80% of the time series</li>
                <li><strong>Testing Data:</strong> Last 20% of the time series</li>
                <li><strong>Model Parameters:</strong> ARIMA(p={}, d=1, q=1)</li>
            </ul>
        </div>
    """.format(p_value), unsafe_allow_html=True)

def create_rbf_interpolation(df):
    st.subheader("RBF-NN Interpolation")
    df = df.rename(columns={'Item': 'Crops'})
    selected_crop = st.selectbox("Select Crop for Interpolation", df['Crops'].unique())
    x_axis = st.selectbox("Select X-axis Variable", 
                         ['year', 'average_rain_fall_mm_per_year', 'avg_temp', 'tonnes_of_active_ingredients'])
    
    L = st.slider("Select L (Length Scale) Parameter", 0.1, 10.0, 1.0, 0.1)
    test_size = st.slider("Select Test Data Size (Percentage)", 0.1, 0.9, 0.2, 0.1)
    crop_data = df[df['Crops'] == selected_crop].dropna(subset=[x_axis, 'yield'])
    x_data = crop_data[x_axis].values
    y_data = crop_data['yield'].values
    # Sort data
    sorted_indices = np.argsort(x_data)
    x_data = x_data[sorted_indices]
    y_data = y_data[sorted_indices]
    # Split data
    split_index = int(len(x_data) * (1 - test_size))
    x_train, x_test = x_data[:split_index], x_data[split_index:]
    y_train, y_test = y_data[:split_index], y_data[split_index:]
    # RBF calculations
    X_train = np.array([[np.exp(-((x_i - x_j) ** 2) / (2 * L ** 2)) 
                         for x_j in x_train] for x_i in x_train])
    U, S, Vt = np.linalg.svd(X_train)
    
    # Compute pseudo-inverse and weights
    S_inv = np.zeros_like(X_train.T)
    S_inv[np.where(S > 1e-10)[0], np.where(S > 1e-10)[0]] = 1 / S[S > 1e-10]
    X_pseudo_inv = Vt.T @ S_inv @ U.T
    w = X_pseudo_inv @ y_train
    # Prepare prediction data
    x_actual = np.linspace(x_data.min(), x_data.max(), 200)
    y_pred = np.array([np.sum([w_c * np.exp(-((x - x_c) ** 2) / (2 * L ** 2)) 
                              for w_c, x_c in zip(w, x_train)]) for x in x_actual])
    # Fit trend line
    coeffs = np.polyfit(x_train, y_train, deg=1)
    trend_line = np.polyval(coeffs, x_actual)
    # Create plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_train, y=y_train, mode='markers', 
                            name='Training Data', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=x_test, y=y_test, mode='markers', 
                            name='Testing Data', marker=dict(color='red')))
    fig.add_trace(go.Scatter(x=x_actual, y=y_pred, mode='lines', 
                            name=f'RBF-NN Prediction (L={L})', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=x_actual, y=trend_line, mode='lines', 
                            name='Trend Line', line=dict(color='black', dash='dash')))
    fig.update_layout(title=f"RBF-NN Interpolation for {selected_crop}",
                     xaxis_title=x_axis,
                     yaxis_title='Yield')
    st.plotly_chart(fig, use_container_width=True)
    add_chart_usage_instructions()
    # Parameters explanation
    st.markdown("""
        <div class='text-area'>
            <h4>Parameters Explanation:</h4>
            <ul>
                <li><strong>L (Length Scale):</strong> Controls the smoothness of the interpolation. Smaller L makes the interpolation more sensitive to nearby points.</li>
                <li><strong>X-axis Variable:</strong> The feature used as input to the RBF-NN model.</li>
                <li><strong>Test Data Size:</strong> Percentage of data used for testing the model.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def create_technical_visualization(datasets):
    st.subheader("Dataset Visualization")
    selected_dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[selected_dataset]

    if len(df.columns) < 2:
        st.write("Not enough columns for plotting.")
        return

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis", df.columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis", df.columns)

    fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{selected_dataset}: {y_axis} vs {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

def plot_missing_heatmap(df, value_column, title, agg_method='first'):
    required_columns = ['Area', 'year', value_column]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Required column '{col}' not found in the DataFrame for {title}")
            return

    df_subset = df[required_columns].copy()
    df_subset = df_subset.groupby(['Area', 'year'])[value_column].agg(agg_method).reset_index()
    pivot = df_subset.pivot(index='year', columns='Area', values=value_column)

    plt.figure(figsize=(20, 10))
    sns.heatmap(pivot.isnull(), cbar=False, cmap='plasma', xticklabels=True, yticklabels=True)
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Area', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.xticks(rotation=90, ha='right')
    num_years = pivot.shape[0]
    y_tick_locations = np.linspace(0, num_years - 1, min(20, num_years)).astype(int)
    y_tick_labels = pivot.index[y_tick_locations]
    plt.yticks(y_tick_locations, y_tick_labels)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def visualize_missing_values(datasets):
    st.subheader("Missing Values Analysis")
    
    datasets_to_plot = {
        'Yield Data': ('yield', 'yield'),
        'Pesticide Data': ('pesticides', 'Value'),
        'Temperature Data': ('temp', 'avg_temp'),
        'Rainfall Data': ('rainfall', 'average_rain_fall_mm_per_year')
    }

    for title, (dataset_key, value_column) in datasets_to_plot.items():
        st.markdown(f"### Missing Values Distribution in {title}")
        plot_missing_heatmap(datasets[dataset_key], value_column, 
                           f"Missing Values Distribution in {title}")

    st.markdown("""
        <div class='text-area' style='text-align: left;'>
            <h4>Missing Values Handling Strategy:</h4>
            <p>
            The heatmaps above show the distribution of missing values across different areas and years (yellow indicates missing values).
            Missing values handling strategy:
            <ol>
                <li>Filter common areas that exist in all datasets and remove other areas.</li>
                <li>Restrict all datasets to the common year range from 1990 to 2013.</li>
                <li>Fill the remaining missing values using MICE (Multiple Imputation by Chained Equations).</li>
            </ol>
            </p>
        </div>
    """, unsafe_allow_html=True)

def mice_comparison_tab(datasets):
    st.subheader("MICE Imputation and Comparison")
    
    selected_dataset = st.selectbox("Select Dataset for MICE Imputation", 
                                  ['yield', 'pesticides'])
    
    value_column = 'yield' if selected_dataset == 'yield' else 'Value'
    df_cleaned = datasets[selected_dataset].copy()
    
    # Perform MICE analysis
    filled_df = iterative_mice_imputation(df_cleaned, value_column)
    compare_datasets(df_cleaned, filled_df, value_column, selected_dataset)

def create_yield_prediction_interface(agricultural_df):
    st.subheader("Predict Yield (Using Agricultural Dataset)")

    required_features = ['year', 'average_rain_fall_mm_per_year', 'tonnes_of_active_ingredients', 'avg_temp', 'yield']
    for col in required_features:
        if col not in agricultural_df.columns:
            st.write(f"Missing required column: {col}")
            return

    crop_list = agricultural_df['Crops'].dropna().unique() if 'Crops' in agricultural_df.columns else []
    area_list = agricultural_df['Area'].dropna().unique() if 'Area' in agricultural_df.columns else []

    if not len(crop_list) or not len(area_list):
        st.write("Crops or Area column not found or empty.")
        return

    selected_crop = st.selectbox("Select Crop for Prediction", crop_list)
    selected_area = st.selectbox("Select Area for Prediction", area_list)

    filtered_df = agricultural_df[(agricultural_df['Crops'] == selected_crop) & 
                                (agricultural_df['Area'] == selected_area)].dropna(subset=required_features)
    if filtered_df.empty:
        st.write("No valid data available for the selected crop and area.")
        return

    X = filtered_df[['year', 'average_rain_fall_mm_per_year', 'tonnes_of_active_ingredients', 'avg_temp']]
    y = filtered_df['yield']

    model = LinearRegression()
    model.fit(X, y)

    st.markdown("### Input Conditions")
    # Modified year input to start from 2014
    input_year = st.number_input("Year", min_value=2014, max_value=2030, value=2014)
    input_rainfall = st.number_input("Average Rainfall (mm/year)",
                                     float(X['average_rain_fall_mm_per_year'].min()),
                                     float(X['average_rain_fall_mm_per_year'].max()),
                                     float(X['average_rain_fall_mm_per_year'].mean()))
    input_pesticide = st.number_input("Tonnes of Active Ingredients (Pesticides)",
                                      float(X['tonnes_of_active_ingredients'].min()),
                                      float(X['tonnes_of_active_ingredients'].max()),
                                      float(X['tonnes_of_active_ingredients'].mean()))
    input_temp = st.number_input("Average Temperature",
                                 float(X['avg_temp'].min()),
                                 float(X['avg_temp'].max()),
                                 float(X['avg_temp'].mean()))

    future_year = st.slider("Forecast up to year (for trend)", 2014, 2030, 2030, step=1)

    if st.button("Predict Yield"):
        input_data = pd.DataFrame({
            'year': [input_year],
            'average_rain_fall_mm_per_year': [input_rainfall],
            'tonnes_of_active_ingredients': [input_pesticide],
            'avg_temp': [input_temp]
        })
        prediction = model.predict(input_data)[0]

        st.markdown(f"""
            <div style="
                background: #e8f5e9;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                border: 2px solid {THEME_COLORS['primary']};
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-top: 20px;">
                <h3 style="color:{THEME_COLORS['primary']};">Predicted Yield</h3>
                <p style="font-size:1.5rem; font-weight:bold; color:{THEME_COLORS['accent']};">
                    {prediction:.2f} units
                </p>
            </div>
        """, unsafe_allow_html=True)

        historical_years = filtered_df['year'].values
        historical_yields = filtered_df['yield'].values
        max_year = int(X['year'].max())
        future_years = np.arange(max_year+1, future_year+1)
        future_predictions = []

        for fy in future_years:
            future_input = pd.DataFrame({
                'year': [fy],
                'average_rain_fall_mm_per_year': [input_rainfall],
                'tonnes_of_active_ingredients': [input_pesticide],
                'avg_temp': [input_temp]
            })
            future_pred = model.predict(future_input)[0]
            future_predictions.append(future_pred)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_years, y=historical_yields, mode='lines+markers',
                                 name='Historical Yield', line=dict(color='blue')))
        if len(future_predictions) > 0:
            fig.add_trace(go.Scatter(x=future_years, y=future_predictions, mode='lines+markers',
                                     name='Future Prediction', line=dict(color='red')))

        fig.add_trace(go.Scatter(x=[input_year], y=[prediction], mode='markers', name='Selected Year Prediction',
                                 marker=dict(size=10, color='green', symbol='star')))

        fig.update_layout(
            title=f"Yield Trend for {selected_crop} in {selected_area} from Historical to {future_year}",
            xaxis_title="Year",
            yaxis_title="Yield",
            legend_title="Legend"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_missing_value_analysis_after_cleaning(df_yield_cleaned, df_pesticides_cleaned, df_temp_cleaned, df_rainfall_cleaned):
    st.markdown("### Missing Values After Cleaning (before using MICE)")
    # For yield, use 'yield' instead of 'Value'
    plot_missing_heatmap(df_yield_cleaned, 'yield', 'Missing Values Distribution in Yield Data (Cleaned)')
    plot_missing_heatmap(df_pesticides_cleaned, 'Value', 'Missing Values Distribution in Pesticide Data (Cleaned)')
    plot_missing_heatmap(df_temp_cleaned, 'avg_temp', 'Missing Values Distribution in Temperature Data (Cleaned)')
    plot_missing_heatmap(df_rainfall_cleaned, 'average_rain_fall_mm_per_year', 'Missing Values Distribution in Rainfall Data (Cleaned)')

def customer_analysis(datasets):
    st.title("Customer Analysis Dashboard")

    tabs = st.tabs([
        "Custom Visualization",
        "Correlation Analysis",
        "Geographic Analysis",
        "PCA Analysis",
        "ARIMA Forecast",
        "RBF-NN Interpolation",
        "Predict Yield"
    ])

    agricultural_data = datasets['agricultural']

    with tabs[0]:
        create_customer_visualization(agricultural_data)

    with tabs[1]:
        create_correlation_heatmap(agricultural_data)

    with tabs[2]:
        create_geographic_time_series(agricultural_data)

    with tabs[3]:
        create_pca_analysis(agricultural_data)

    with tabs[4]:
        create_arima_forecast(agricultural_data)

    with tabs[5]:
        create_rbf_interpolation(agricultural_data)

    with tabs[6]:
        create_yield_prediction_interface(agricultural_data)
        
def create_yearly_counts_plot(original_df, filled_df, ax, dataset_name='Dataset'):
    original_counts = original_df.groupby('year').size()
    filled_counts = filled_df.groupby('year').size()

    x = np.arange(len(original_counts.index))
    width = 0.35

    ax.bar(x - width/2, original_counts.values, width, label='Before Imputation', color='blue', alpha=0.7)
    ax.bar(x + width/2, filled_counts.values, width, label='After Imputation', color='orange', alpha=0.7)

    ax.set_title(f'{dataset_name.capitalize()} Data Counts by Year', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Observations')
    ax.set_xticks(x)
    ax.set_xticklabels(original_counts.index, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def create_time_trends_plot(original_df, filled_df, value_column='Value', ax=None, dataset_name='Dataset'):
    orig_means = original_df.groupby('year')[value_column].mean().reset_index()
    filled_means = filled_df.groupby('year')[value_column].mean().reset_index()

    ax.plot(orig_means['year'], orig_means[value_column], marker='o', color='blue', label='Before Imputation Mean')
    ax.plot(filled_means['year'], filled_means[value_column], marker='s', linestyle='--', color='orange', label='After Imputation Mean')
    ax.set_title(f'{dataset_name.capitalize()} {value_column} Trends Over Time', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel(value_column)
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

def compare_datasets(original_df, filled_df, value_column='Value', dataset_name='Dataset'):
    if value_column not in original_df.columns or value_column not in filled_df.columns:
        st.error(f"Column '{value_column}' not found in one of the datasets.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Distribution Plot
    sns.histplot(data=original_df, x=value_column, bins=50, color='blue', alpha=0.5, 
                label='Before Imputation', stat='density', ax=axes[0])
    sns.histplot(data=filled_df, x=value_column, bins=50, color='orange', alpha=0.5, 
                label='After Imputation', stat='density', ax=axes[0])
    axes[0].set_title(f'{dataset_name.capitalize()} Value Distribution Comparison', fontsize=14)
    axes[0].set_xlabel(value_column)
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Yearly Counts Plot
    create_yearly_counts_plot(original_df, filled_df, ax=axes[1], dataset_name=dataset_name)

    # Time Trends Plot
    create_time_trends_plot(original_df, filled_df, value_column=value_column, ax=axes[2], dataset_name=dataset_name)

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()
    
def technical_details(datasets):
    st.title("Technical Analysis Dashboard")

    tech_tabs = st.tabs([
        "Dataset Visualization",
        "Missing Values Analysis",
        "MICE Comparison"
    ])

    with tech_tabs[0]:
        create_technical_visualization(datasets)

    with tech_tabs[1]:
        visualize_missing_values(datasets)
        
        # Data cleaning steps
        df_yield = datasets['yield']
        df_pesticides = datasets['pesticides']
        df_temp = datasets['temp']
        df_rainfall = datasets['rainfall']

        # Perform data cleaning
        df_yield_dropped = df_yield.dropna(thresh=df_yield.shape[1]*0.5)
        df_pesticides_dropped = df_pesticides.dropna(thresh=df_pesticides.shape[1]*0.5)
        df_temp_dropped = df_temp.dropna(thresh=df_temp.shape[1]*0.5)
        df_rainfall_dropped = df_rainfall.dropna(thresh=df_rainfall.shape[1]*0.5)

        # Find common areas
        common_areas = set(df_yield_dropped['Area']) \
            .intersection(set(df_pesticides_dropped['Area'])) \
            .intersection(set(df_temp_dropped['Area'])) \
            .intersection(set(df_rainfall_dropped['Area']))

        # Filter data
        yield_data_filtered = df_yield_dropped[df_yield_dropped['Area'].isin(common_areas)]
        pesticides_data_filtered = df_pesticides_dropped[df_pesticides_dropped['Area'].isin(common_areas)]
        df_temp_filtered = df_temp_dropped[df_temp_dropped['Area'].isin(common_areas)]
        df_rainfall_filtered = df_rainfall_dropped[df_rainfall_dropped['Area'].isin(common_areas)]

        # Time range filtering
        df_yield_cleaned = yield_data_filtered[(yield_data_filtered['year'] >= 1990) & (yield_data_filtered['year'] <= 2013)]
        df_pesticides_cleaned = pesticides_data_filtered[(pesticides_data_filtered['year'] >= 1990) & (pesticides_data_filtered['year'] <= 2013)]
        df_temp_cleaned = df_temp_filtered[(df_temp_filtered['year'] >= 1990) & (df_temp_filtered['year'] <= 2013)]
        df_rainfall_cleaned = df_rainfall_filtered[(df_rainfall_filtered['year'] >= 1990) & (df_rainfall_filtered['year'] <= 2013)]

        show_missing_value_analysis_after_cleaning(df_yield_cleaned, df_pesticides_cleaned, df_temp_cleaned, df_rainfall_cleaned)

    with tech_tabs[2]:
        mice_comparison_tab(datasets, df_yield_cleaned, df_pesticides_cleaned)

def iterative_mice_imputation(df, value_column='Value'):
    pivot_df = df.pivot_table(index='Area', columns='year', 
                             values=value_column, aggfunc='first')
    
    imputer = IterativeImputer(max_iter=10, random_state=42, min_value=0)
    imputed_values = imputer.fit_transform(pivot_df)
    
    imputed_df = pd.DataFrame(
        imputed_values, 
        index=pivot_df.index, 
        columns=pivot_df.columns
    ).reset_index()
    
    imputed_df = imputed_df.melt(id_vars='Area', 
                                var_name='year', 
                                value_name=value_column)
    imputed_df['year'] = imputed_df['year'].astype(int)
    imputed_df[value_column] = imputed_df[value_column].round()
    
    return imputed_df

def mice_comparison_tab(datasets, df_yield_cleaned, df_pesticides_cleaned):
    st.subheader("MICE Imputation and Comparison")
    
    selected_dataset = st.selectbox("Select Dataset for MICE Imputation", 
                                  ['pesticides', 'yield'])
    
    # Use cleaned datasets instead of raw datasets
    if selected_dataset == 'yield':
        value_column = 'yield'
        df_cleaned = df_yield_cleaned.copy()
    else:
        value_column = 'Value'
        df_cleaned = df_pesticides_cleaned.copy()
    
    # Perform MICE analysis
    filled_df = iterative_mice_imputation(df_cleaned, value_column)
    compare_datasets(df_cleaned, filled_df, value_column, selected_dataset)
    
def create_yearly_counts_plot(original_df, filled_df, ax, dataset_name='Dataset'):
    original_counts = original_df.groupby('year').size()
    filled_counts = filled_df.groupby('year').size()

    x = np.arange(len(original_counts.index))
    width = 0.35

    ax.bar(x - width/2, original_counts.values, width, label='Before Imputation', color='blue', alpha=0.7)
    ax.bar(x + width/2, filled_counts.values, width, label='After Imputation', color='orange', alpha=0.7)

    ax.set_title(f'{dataset_name.capitalize()} Data Counts by Year', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Observations')
    ax.set_xticks(x)
    ax.set_xticklabels(original_counts.index, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def create_time_trends_plot(original_df, filled_df, value_column='Value', ax=None, dataset_name='Dataset'):
    orig_means = original_df.groupby('year')[value_column].mean().reset_index()
    filled_means = filled_df.groupby('year')[value_column].mean().reset_index()

    ax.plot(orig_means['year'], orig_means[value_column], marker='o', color='blue', label='Before Imputation Mean')
    ax.plot(filled_means['year'], filled_means[value_column], marker='s', linestyle='--', color='orange', label='After Imputation Mean')
    ax.set_title(f'{dataset_name.capitalize()} {value_column} Trends Over Time', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel(value_column)
    ax.legend()
    ax.grid(True)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))

def compare_datasets(original_df, filled_df, value_column='Value', dataset_name='Dataset'):
    if value_column not in original_df.columns or value_column not in filled_df.columns:
        st.error(f"Column '{value_column}' not found in one of the datasets.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Distribution Plot
    sns.histplot(data=original_df, x=value_column, bins=50, color='blue', alpha=0.5, 
                label='Before Imputation', stat='density', ax=axes[0])
    sns.histplot(data=filled_df, x=value_column, bins=50, color='orange', alpha=0.5, 
                label='After Imputation', stat='density', ax=axes[0])
    axes[0].set_title(f'{dataset_name.capitalize()} Value Distribution Comparison', fontsize=14)
    axes[0].set_xlabel(value_column)
    axes[0].set_ylabel('Density')
    axes[0].legend()

    # Yearly Counts Plot
    create_yearly_counts_plot(original_df, filled_df, ax=axes[1], dataset_name=dataset_name)

    # Time Trends Plot
    create_time_trends_plot(original_df, filled_df, value_column=value_column, ax=axes[2], dataset_name=dataset_name)

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()

def main():
    with st.spinner("Loading data..."):
        datasets = load_data()

    img_str = load_and_process_bg_image()
    apply_styling(img_str)

    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Select Section", ["Home", "Customer Analysis", "Technical Details"])

    if section == "Home":
        show_homepage()
        st.markdown("""
            <div class='text-area'>
                <h2>Available Datasets</h2>
                <p>Our analysis is based on the following datasets from <a href="https://www.kaggle.com/code/patelris/crop-yield-eda-viz/input" target="_blank">Kaggle: Crop Yield EDA & Visualization</a>:</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class='text-area'>
                    <h4>Primary Dataset:</h4>
                    <p><strong>Agricultural Dataset:</strong> Includes:</p>
                    <ul>
                        <li>Crop yields</li>
                        <li>Rainfall data</li>
                        <li>Pesticide usage (tonnes_of_active_ingredients)</li>
                        <li>Temperature data (avg_temp)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                <div class='text-area'>
                    <h4>Supporting Datasets:</h4>
                    <ul>
                        <li>Detailed rainfall records</li>
                        <li>Temperature history</li>
                        <li>Pesticide application data</li>
                        <li>Historical yield information</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

    elif section == "Customer Analysis":
        customer_analysis(datasets)

    elif section == "Technical Details":
        technical_details(datasets)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the input data and try again.")
