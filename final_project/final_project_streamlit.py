import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import base64
from PIL import Image
import requests
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.ticker as mticker
import logging

# Configure logging and page
logging.basicConfig(level=logging.ERROR)
st.set_page_config(page_title="CropInsight", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# Constants
DATASET_DESCRIPTIONS = [
    """
    <div style='margin-left: 20px;'>
        <p><strong>Comprehensive dataset combining information from all sources:</strong></p>
        <ul style='list-style-type: disc; margin-left: 20px;'>
            <li><strong>Crops:</strong> Rice, Wheat, Maize, Soybeans</li>
            <li><strong>Time Period:</strong> 1990-2013</li>
            <li><strong>Variables:</strong>
                <ul style='list-style-type: circle; margin-left: 20px;'>
                    <li>Yield measurements</li>
                    <li>Pesticide usage</li>
                    <li>Temperature data</li>
                    <li>Rainfall information</li>
                </ul>
            </li>
            <li><strong>Regions:</strong> Multiple countries and regions</li>
        </ul>
    </div>
    """,
    """
    <div style='margin-left: 20px;'>
        <p><strong>Contains crop yield data for different regions:</strong></p>
        <ul style='list-style-type: disc; margin-left: 20px;'>
            <li><strong>Key Features:</strong>
                <ul style='list-style-type: circle; margin-left: 20px;'>
                    <li>Crop types and their yields</li>
                    <li>Annual yield measurements</li>
                    <li>Regional yield variations</li>
                </ul>
            </li>
            <li><strong>Units:</strong> Tonnes per hectare</li>
            <li><strong>Geographical Coverage:</strong> Multiple regions</li>
        </ul>
    </div>
    """,
    """
    <div style='margin-left: 20px;'>
        <p><strong>Information about pesticide usage:</strong></p>
        <ul style='list-style-type: disc; margin-left: 20px;'>
            <li><strong>Measurements:</strong>
                <ul style='list-style-type: circle; margin-left: 20px;'>
                    <li>Pesticide application rates</li>
                    <li>Types of pesticides</li>
                    <li>Regional usage patterns</li>
                </ul>
            </li>
            <li><strong>Units:</strong> Tonnes of active ingredients</li>
            <li><strong>Time Scale:</strong> Annual data</li>
        </ul>
    </div>
    """,
    """
    <div style='margin-left: 20px;'>
        <p><strong>Regional temperature records:</strong></p>
        <ul style='list-style-type: disc; margin-left: 20px;'>
            <li><strong>Measurements:</strong>
                <ul style='list-style-type: circle; margin-left: 20px;'>
                    <li>Average temperatures</li>
                    <li>Regional variations</li>
                    <li>Temporal patterns</li>
                </ul>
            </li>
            <li><strong>Units:</strong> Degrees Celsius</li>
            <li><strong>Temporal Resolution:</strong> Annual averages</li>
        </ul>
    </div>
    """,
    """
    <div style='margin-left: 20px;'>
        <p><strong>Precipitation data across regions:</strong></p>
        <ul style='list-style-type: disc; margin-left: 20px;'>
            <li><strong>Measurements:</strong>
                <ul style='list-style-type: circle; margin-left: 20px;'>
                    <li>Annual rainfall amounts</li>
                    <li>Regional precipitation patterns</li>
                    <li>Seasonal variations</li>
                </ul>
            </li>
            <li><strong>Units:</strong> Millimeters per year</li>
            <li><strong>Coverage:</strong> Regional rainfall data</li>
        </ul>
    </div>
    """
]

# Cache data loading
@st.cache_data
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
            df = df.drop(columns=["Domain Code", "Domain", "Area Code", "Element Code", "Item Code", "Element", "Year Code", "Unit"], errors='ignore')
            df.rename(columns={'Year': 'year', 'Item': 'Crops', 'Value': 'yield'}, inplace=True)
        elif key == 'pesticides':
            df = df.drop(columns=["Domain", "Item", "Element", "Unit"], errors='ignore')
            df.rename(columns={'Year': 'year'}, inplace=True)
            # Keep 'Value' column as is in 'pesticides' dataset
        else:
            df.rename(columns={'Year': 'year'}, inplace=True)
        data[key] = df
    return data

def load_and_process_bg_image():
    try:
        response = requests.get("https://img.freepik.com/free-photo/detail-rice-plant-sunset-valencia-with-plantation-out-focus-rice-grains-plant-seed_181624-25838.jpg")
        image = Image.open(BytesIO(response.content))
        image = image.resize((1920, 1080))
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        st.error(f"Error loading background image: {str(e)}")
        return None

def apply_styling(img_str):
    background_style = f"""
        background-image: url("data:image/jpg;base64,{img_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    """ if img_str else "background-color: #f5f5f5;"

    st.markdown(f"""
        <style>
        .stApp {{
            {background_style}
        }}
        
        /* Base styles */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.85);
            z-index: -1;
        }}

        /* Component containers */
        .text-area {{
            background: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}

        .info-card {{
            background: rgba(255, 255, 255, 0.9);
            padding: 1rem;
            margin-bottom: 1rem;
            border-radius: 8px;
        }}

        /* Typography */
        h1, h2, h3 {{
            color: #2C3E50;
            font-weight: bold;
            margin-bottom: 1rem;
        }}

        h1 {{
            font-size: 3rem;
            font-family: "Georgia", serif;
            text-align: center;
        }}

        h2 {{
            font-size: 2.2rem;
            color: #2980B9;
            text-align: center;
        }}

        h3 {{
            font-size: 1.8rem;
            color: #2980B9;
        }}

        p {{
            font-size: 1.1rem;
            line-height: 1.6;
            color: #2C3E50;
            margin-bottom: 1rem;
        }}

        /* Lists */
        ul {{
            list-style-position: inside;
            padding-left: 1rem;
        }}

        li {{
            margin: 0.8rem 0;
            line-height: 1.6;
            color: #2C3E50;
        }}

        /* Input elements */
        div[data-baseweb="select"],
        .stSelectbox select {{
            background-color: white;
            border-radius: 4px;
            margin-bottom: 15px;
            padding: 2px;
        }}

        .stSelectbox label {{
            color: #2C3E50;
            font-weight: bold;
            font-size: 1.1rem;
            margin-bottom: 5px;
        }}

        /* Tabs */
        div[data-testid="stHorizontalBlock"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 5px;
        }}

        div[data-testid="stHorizontalBlock"] button {{
            background-color: rgba(255, 255, 255, 0.9);
            color: #2C3E50;
            border: none;
            border-radius: 5px;
            margin-right: 5px;
            padding: 10px;
            font-weight: bold;
        }}

        div[data-testid="stHorizontalBlock"] button:hover,
        div[data-testid="stHorizontalBlock"] button:focus,
        div[data-testid="stHorizontalBlock"] button[aria-selected="true"] {{
            background-color: #2980B9;
            color: white;
            outline: none;
        }}

        </style>
    """, unsafe_allow_html=True)

def add_chart_usage_instructions():
    st.markdown("""
        <div style='
            background-color: #f0f2f6; 
            padding: 15px; 
            border-radius: 8px; 
            margin: 30px 0; 
            text-align: center; 
            border: 2px solid #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        '>
            <p style='
                margin: 0; 
                color: #1f77b4; 
                font-weight: bold;
                font-size: 1.2rem;
                letter-spacing: 0.5px;
            '>
                📊 <span style='text-decoration: underline;'>Chart Usage:</span> 
                Click on a legend item to hide it; double-click to isolate it
            </p>
        </div>
    """, unsafe_allow_html=True)

def show_homepage(datasets):
    st.markdown("""
        <div class='text-area'>
            <h1>🌾 CropInsight: Enhancing Agricultural Productivity</h1>
        </div>
        <div class='text-area'>
            <h2>Welcome to CropInsight</h2>
            <p>
            CropInsight aims to enhance agricultural productivity by providing tools that analyze how various factors affect crop yields.
            This platform is tailored for agricultural professionals, data scientists, policymakers, and researchers seeking to make informed decisions based on data-driven insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    create_understanding_section()
    show_dataset_information(datasets)

    st.markdown("""
        <div class='text-area'>
            <h2>Let's Explore Together!</h2>
            <p style='font-size:0.9rem;'>
            Please note that due to the size of the datasets, some analyses may take longer to update.
            We appreciate your patience as we work to provide you with insightful results.
            </p>
        </div>
    """, unsafe_allow_html=True)

def create_understanding_section():
    st.markdown("""
        <div class='text-area'>
            <h2>Understanding CropInsight</h2>
        </div>
    """, unsafe_allow_html=True)

    content = {
        "Why": "To improve agricultural productivity by understanding how various factors affect crop yields.",
        "Who": "Agricultural professionals, data scientists, policymakers, and researchers.",
        "What": "CropInsight offers tools for analyzing agricultural data through visualizations and predictive models.",
        "When": "Useful for planning new agricultural cycles and conducting research related to climate impacts on crops.",
        "Where": "Data is sourced from various agricultural datasets, including yield statistics and climate information.",
        "How": "By combining historical data analysis with advanced visualization techniques and predictive modeling."
    }

    for key, value in content.items():
        st.markdown(f"""
            <div class='info-card'>
                <h3>{key}</h3>
                <p>{value}</p>
            </div>
        """, unsafe_allow_html=True)

def show_dataset_information(datasets):
    st.markdown("""
        <div class='text-area'>
            <h2>Available Datasets</h2>
        </div>
    """, unsafe_allow_html=True)

    dataset_tabs = st.tabs([
        "Agricultural Dataset",
        "Yield Dataset",
        "Pesticides Dataset",
        "Temperature Dataset",
        "Rainfall Dataset"
    ])

    datasets_list = ['agricultural', 'yield', 'pesticides', 'temp', 'rainfall']
    titles = [
        "Agricultural Dataset (Integrated)",
        "Yield Dataset",
        "Pesticides Dataset",
        "Temperature Dataset",
        "Rainfall Dataset"
    ]

    for idx, tab in enumerate(dataset_tabs):
        with tab:
            st.markdown(f"""
                <div class='text-area'>
                    <h3>{titles[idx]}</h3>
                    {DATASET_DESCRIPTIONS[idx]}
                </div>
            """, unsafe_allow_html=True)
            
            df = datasets[datasets_list[idx]]
            st.markdown(f"""
                <div style='text-align: center;'>
                    <h4>{titles[idx]} Data Preview</h4>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("<div class='dataframe-container'>", unsafe_allow_html=True)
            st.dataframe(df.head(10))
            st.markdown("</div>", unsafe_allow_html=True)

def create_visualization_plot(df, x_axis, y_axis, color_by=None, title=None):
    """Common function for creating visualization plots"""
    fig = px.scatter(df, 
                    x=x_axis, 
                    y=y_axis, 
                    color=df[color_by] if color_by else None,
                    title=title)
    st.plotly_chart(fig, use_container_width=True)
    add_chart_usage_instructions()

def create_customer_visualization(df):
    st.subheader("Scatter Plot")
    df = df.rename(columns={'Item': 'Crops'})

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis", df.select_dtypes(include=[np.number]).columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis", df.select_dtypes(include=[np.number]).columns)

    color_by = st.selectbox("Select Color By", ['Crops', 'Area'])
    
    create_visualization_plot(df, x_axis, y_axis, color_by, f"{y_axis} vs {x_axis} by {color_by}")

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
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        hovertemplate='Variable 1: %{x}<br>Variable 2: %{y}<br>Correlation: %{z}<extra></extra>',
    ))

    fig.update_layout(
        title=f"Correlation Heatmap for {selected_crop}",
        xaxis_title="",
        yaxis_title="",
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_tickangle=45,
        yaxis_autorange='reversed',
        width=700,
        height=700,
    )

    st.plotly_chart(fig, use_container_width=True)
    # No chart usage instructions here

def create_geographic_time_series(df):
    st.subheader("Geographic Analysis")
    df = df.rename(columns={'Item': 'Crops'})

    year = st.slider("Select Year",
                     min_value=int(df['year'].min()),
                     max_value=int(df['year'].max()))

    selected_crop = st.selectbox("Select Crop for Map", df['Crops'].unique())
    year_data = df[(df['year'] == year) & (df['Crops'] == selected_crop)]

    fig = px.scatter_geo(year_data,
                        locations="Area",
                        locationmode="country names",
                        color="yield",
                        hover_name="Area",
                        size="yield",
                        title=f"{selected_crop} Yield by Region ({year})")
    st.plotly_chart(fig, use_container_width=True)
    # No chart usage instructions here

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
    df = df.rename(columns={'Item': 'Crops'})

    selected_crop = st.selectbox("Select Crop for Forecast", df['Crops'].unique())
    p_value = st.slider("Select AR(p) Parameter", 1, 5, 2)
    test_size = st.slider("Select Test Data Size", 1, 10, 5)

    crop_data = df[df['Crops'] == selected_crop]['yield'].values
    train_data, test_data = crop_data[:-test_size], crop_data[-test_size:]

    model = ARIMA(train_data, order=(p_value,1,1))
    results = model.fit()
    forecast = results.forecast(steps=test_size)

    # Training and testing data plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_data, name="Training Data", mode='lines'))
    fig.add_trace(go.Scatter(x=np.arange(len(train_data), len(train_data) + len(test_data)), 
                            y=test_data, name="Testing Data", mode='lines'))
    fig.update_layout(title=f"Training and Testing Data for {selected_crop}",
                     xaxis_title='Time',
                     yaxis_title='Yield')
    st.plotly_chart(fig, use_container_width=True)
    add_chart_usage_instructions()

    # Forecast plot
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(y=test_data, name="Actual Testing Data", mode='lines'))
    fig_forecast.add_trace(go.Scatter(y=forecast, name="Forecast", mode='lines'))
    fig_forecast.update_layout(title=f"Forecast vs Actual Testing Data for {selected_crop}",
                             xaxis_title='Time',
                             yaxis_title='Yield')
    st.plotly_chart(fig_forecast, use_container_width=True)
    add_chart_usage_instructions()

    st.markdown("""
        <div class='text-area'>
            <h4>Parameters Explanation:</h4>
            <ul>
                <li><strong>AR(p) Parameter:</strong> The number of lag observations included in the model. Higher values can capture more complex patterns.</li>
                <li><strong>Test Data Size:</strong> The number of data points used for testing the model.</li>
            </ul>
            <p><em>Note: The model uses differencing (d=1) and moving average (q=1) by default to handle trends and seasonal patterns.</em></p>
        </div>
    """, unsafe_allow_html=True)

def create_rbf_interpolation(df):
    st.subheader("RBF-NN Interpolation")
    df = df.rename(columns={'Item': 'Crops'})

    selected_crop = st.selectbox("Select Crop for Interpolation", df['Crops'].unique())
    x_axis = st.selectbox("Select X-axis Variable", 
                         ['year', 'average_rain_fall_mm_per_year', 'avg_temp', 'pesticides'])
    
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
        
def customer_analysis(datasets):
    st.title("Customer Analysis Dashboard")
    
    tabs = st.tabs([
        "Custom Visualization",
        "Correlation Analysis",
        "Geographic Analysis",
        "PCA Analysis",
        "ARIMA Forecast",
        "RBF-NN Interpolation"
    ])

    agricultural_data = datasets['agricultural']

    with tabs[0]: create_customer_visualization(agricultural_data)
    with tabs[1]: create_correlation_heatmap(agricultural_data)
    with tabs[2]: create_geographic_time_series(agricultural_data)
    with tabs[3]: create_pca_analysis(agricultural_data)
    with tabs[4]: create_arima_forecast(agricultural_data)
    with tabs[5]: create_rbf_interpolation(agricultural_data)

def technical_details(datasets):
    st.title("Technical Analysis Dashboard")
    
    tech_tabs = st.tabs([
        "Dataset Visualization",
        "Missing Values Analysis",
        "MICE Imputation and Comparison"
    ])

    with tech_tabs[0]: create_technical_visualization(datasets)
    with tech_tabs[1]: visualize_missing_values(datasets)
    with tech_tabs[2]: mice_imputation_and_comparison(datasets)

def create_technical_visualization(datasets):
    st.subheader("Dataset Visualization")
    
    selected_dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[selected_dataset]

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("Select X-axis", df.columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis", df.columns)
    with col3:
        color_by = st.selectbox("Select Color By", [None] + list(df.columns))

    create_visualization_plot(df, x_axis, y_axis, color_by, 
                            f"{selected_dataset}: {y_axis} vs {x_axis}")

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

def plot_missing_heatmap(df, value_column, title):
    required_columns = ['year', value_column, 'Area']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Required column '{col}' not found in the DataFrame")
            return

    df_subset = df[required_columns].copy()
    df_subset = df_subset.groupby(['Area', 'year'])[value_column].agg('first').reset_index()
    pivot = df_subset.pivot(index='year', columns='Area', values=value_column)

    plt.figure(figsize=(20, 10))
    sns.heatmap(pivot.isnull(),
                cbar=False,
                cmap='plasma',
                xticklabels=True,
                yticklabels=True)

    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Area', fontsize=12)
    plt.ylabel('Year', fontsize=12)
    plt.xticks(rotation=90, ha='right')

    num_years = pivot.shape[0]
    y_tick_locations = np.linspace(0, num_years-1, min(20, num_years)).astype(int)
    y_tick_labels = pivot.index[y_tick_locations]
    plt.yticks(y_tick_locations, y_tick_labels)

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

def mice_imputation_and_comparison(datasets):
    st.subheader("MICE Imputation and Comparison")
    
    selected_dataset = st.selectbox("Select Dataset for MICE Imputation", 
                                  ['yield', 'pesticides'])
    
    value_column = 'yield' if selected_dataset == 'yield' else 'Value'
    df_cleaned = datasets[selected_dataset].copy()
    analyze_and_impute_data(df_cleaned, value_column, selected_dataset)

def analyze_and_impute_data(original_df, value_column='Value', dataset_name='Dataset'):
    filled_df = iterative_mice_imputation(original_df, value_column)
    compare_datasets(original_df, filled_df, value_column, dataset_name)

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

def compare_datasets(original_df, filled_df, value_column='Value', dataset_name='Dataset'):
    """
    Compare datasets before and after imputation with detailed visualizations and statistics.
    """
    if value_column not in original_df.columns or value_column not in filled_df.columns:
        st.error(f"Column '{value_column}' not found in one of the datasets.")
        return

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Distribution Plot
    sns.histplot(data=original_df, x=value_column, bins=50, color='blue', alpha=0.5, label='Before Imputation', stat='density', ax=axes[0])
    sns.histplot(data=filled_df, x=value_column, bins=50, color='orange', alpha=0.5, label='After Imputation', stat='density', ax=axes[0])
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
    plt.clf()  # Clear the figure after displaying

def create_yearly_counts_plot(original_df, filled_df, ax, dataset_name='Dataset'):
    """
    Plot yearly data counts comparison.
    """
    original_counts = original_df.groupby('year').size()
    filled_counts = filled_df.groupby('year').size()

    x = np.arange(len(original_counts.index))
    width = 0.35

    ax.bar(x - width / 2, original_counts.values, width, label='Before Imputation', color='blue', alpha=0.7)
    ax.bar(x + width / 2, filled_counts.values, width, label='After Imputation', color='orange', alpha=0.7)

    ax.set_title(f'{dataset_name.capitalize()} Data Counts by Year', fontsize=14)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Observations')
    ax.set_xticks(x)
    ax.set_xticklabels(original_counts.index, rotation=45)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

def create_time_trends_plot(original_df, filled_df, value_column='Value', ax=None, dataset_name='Dataset'):
    """
    Plot time trends comparison.
    """
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
    plt.tight_layout()

def main():
    try:
        datasets = load_data()
        img_str = load_and_process_bg_image()
        apply_styling(img_str)

        st.sidebar.title("Navigation")
        section = st.sidebar.radio("Select Section", 
                                 ["Home", "Customer Analysis", "Technical Details"])

        if 'previous_section' not in st.session_state:
            st.session_state['previous_section'] = section
        elif st.session_state['previous_section'] != section:
            st.session_state['previous_section'] = section
            st.markdown(
                """
                <script>
                window.scrollTo(0, 0);
                </script>
                """,
                unsafe_allow_html=True
            )

        if section == "Home":
            show_homepage(datasets)
        elif section == "Customer Analysis":
            customer_analysis(datasets)
        else:
            technical_details(datasets)

    except Exception as e:
        logging.exception("An error occurred.")
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the input data and try again.")

if __name__ == "__main__":
    main()
