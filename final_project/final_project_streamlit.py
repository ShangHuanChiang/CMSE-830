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
from scipy.interpolate import RBFInterpolator
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Page configuration
st.set_page_config(
    page_title="CropInsight",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache data loading
@st.cache_data
def load_data():
    """Load all datasets with caching and drop specified columns"""
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
        # Drop specified columns
        if key == 'yield':
            df = df.drop(columns=["Domain Code", "Domain", "Area Code", "Element Code",
                                  "Item Code", "Element", "Year Code", "Unit"], errors='ignore')
            # Rename 'Year' to 'year' and 'Item' to 'Crops'
            df.rename(columns={'Year': 'year', 'Item': 'Crops'}, inplace=True)
        elif key == 'pesticides':
            df = df.drop(columns=["Domain", "Item", "Element", "Unit"], errors='ignore')
            df.rename(columns={'Year': 'year'}, inplace=True)
        else:
            df.rename(columns={'Year': 'year'}, inplace=True)
        data[key] = df
    return data

def load_and_process_bg_image():
    """Load and process background image with proper handling for text visibility"""
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
    """Apply enhanced styling with background image"""
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
        /* Additional custom styles */
        /* Main container */
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

        /* Text areas and sections */
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

        /* Typography styles */
        h1, h2, h3, p, li, label, .stRadio label {{
            font-family: Arial, sans-serif;
            text-shadow:
                -1px -1px 0 #FFFFFF,
                1px -1px 0 #FFFFFF,
                -1px 1px 0 #FFFFFF,
                1px 1px 0 #FFFFFF;
            background: rgba(255, 255, 255, 0.9);
            padding: 0.5rem;
            border-radius: 8px;
            display: inline-block;
        }}

        /* Additional padding for headers */
        h1 {{
            font-size: 3rem;
            color: #2C3E50;
            font-weight: bold;
            margin-bottom: 2rem;
        }}

        h2 {{
            font-size: 2.2rem;
            color: #34495E;
            font-weight: 600;
            margin: 1.5rem 0;
        }}

        h3 {{
            font-size: 1.8rem;
            color: #2980B9;
            font-weight: 600;
            margin: 1rem 0;
        }}

        p {{
            font-size: 1.1rem;
            line-height: 1.6;
            color: #2C3E50;
            margin-bottom: 1rem;
        }}

        ul {{
            list-style-position: inside;
            padding-left: 1rem;
        }}

        li {{
            margin: 0.8rem 0;
            line-height: 1.6;
            color: #2C3E50;
        }}

        /* Get Started section */
        .get-started-section {{
            background: linear-gradient(135deg, #2980B9, #3498DB);
            color: white;
            padding: 3rem;
            border-radius: 10px;
            text-align: center;
            margin: 3rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }}

        .get-started-section h2,
        .get-started-section p {{
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

def show_homepage():
    """Display homepage with dataset information"""
    st.markdown("""
        <div class='centered-title'>
            <h1 style='color: #2C3E50; font-family: "Georgia", serif;'>🌾 CropInsight: Enhancing Agricultural Productivity</h1>
        </div>

        <div class='text-area'>
            <h2 style='text-align: center; color: #2980B9;'>Welcome to CropInsight</h2>
            <p style='text-align: center; font-size: 1.2rem;'>
            CropInsight aims to enhance agricultural productivity by providing tools that analyze how various factors affect crop yields.
            This platform is tailored for agricultural professionals, data scientists, policymakers, and researchers seeking to make informed decisions based on data-driven insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Add Understanding CropInsight section
    create_understanding_section()

    # Add dataset information
    show_dataset_information()        

def create_understanding_section():
    """Create the 'Understanding CropInsight' section"""
    st.markdown("""
        <div class='section-header'>
            <h2 style='text-align: center; color: #2980B9;'>Understanding CropInsight</h2>
        </div>
    """, unsafe_allow_html=True)

    # Content mapping
    content = {
        "Why": "To improve agricultural productivity by understanding how various factors affect crop yields.",
        "Who": "Agricultural professionals, data scientists, policymakers, and researchers.",
        "What": "CropInsight offers tools for analyzing agricultural data through visualizations and predictive models.",
        "When": "Useful for planning new agricultural cycles and conducting research related to climate impacts on crops.",
        "Where": "Data is sourced from various agricultural datasets, including yield statistics and climate information.",
        "How": "By combining historical data analysis with advanced visualization techniques and predictive modeling."
    }

    # Display all content
    for key, value in content.items():
        st.markdown(f"""
            <div class='info-card'>
                <h3 style='color: #2980B9;'>{key}</h3>
                <p>{value}</p>
            </div>
        """, unsafe_allow_html=True)

def show_dataset_information():
    """Display detailed information about available datasets"""
    st.markdown("""
        <div class='section-header'>
            <h2 style='text-align: center;'>Available Datasets</h2>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs for different datasets
    dataset_tabs = st.tabs([
        "Agricultural Dataset", 
        "Yield Dataset", 
        "Pesticides Dataset",
        "Temperature Dataset", 
        "Rainfall Dataset"
    ])

    with dataset_tabs[0]:
        st.markdown("""
            <div class='text-area'>
                <h3>Agricultural Dataset (Integrated)</h3>
                <p>Comprehensive dataset combining information from all sources:</p>
                <ul>
                    <li><strong>Crops:</strong> Rice, Wheat, Maize, Soybeans</li>
                    <li><strong>Time Period:</strong> 1990-2013</li>
                    <li><strong>Variables:</strong>
                        <ul>
                            <li>Yield measurements</li>
                            <li>Pesticide usage</li>
                            <li>Temperature data</li>
                            <li>Rainfall information</li>
                        </ul>
                    </li>
                    <li><strong>Regions:</strong> Multiple countries and regions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Show sample data
        if st.checkbox("Show Agricultural Dataset Sample"):
            st.dataframe(datasets['agricultural'].head())

    with dataset_tabs[1]:
        st.markdown("""
            <div class='text-area'>
                <h3>Yield Dataset</h3>
                <p>Contains crop yield data for different regions:</p>
                <ul>
                    <li><strong>Key Features:</strong>
                        <ul>
                            <li>Crop types and their yields</li>
                            <li>Annual yield measurements</li>
                            <li>Regional yield variations</li>
                        </ul>
                    </li>
                    <li><strong>Units:</strong> Tonnes per hectare</li>
                    <li><strong>Geographical Coverage:</strong> Multiple regions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        if st.checkbox("Show Yield Dataset Sample"):
            st.dataframe(datasets['yield'].head())

    with dataset_tabs[2]:
        st.markdown("""
            <div class='text-area'>
                <h3>Pesticides Dataset</h3>
                <p>Information about pesticide usage:</p>
                <ul>
                    <li><strong>Measurements:</strong>
                        <ul>
                            <li>Pesticide application rates</li>
                            <li>Types of pesticides</li>
                            <li>Regional usage patterns</li>
                        </ul>
                    </li>
                    <li><strong>Units:</strong> Tonnes of active ingredients</li>
                    <li><strong>Time Scale:</strong> Annual data</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        if st.checkbox("Show Pesticides Dataset Sample"):
            st.dataframe(datasets['pesticides'].head())

    with dataset_tabs[3]:
        st.markdown("""
            <div class='text-area'>
                <h3>Temperature Dataset</h3>
                <p>Regional temperature records:</p>
                <ul>
                    <li><strong>Measurements:</strong>
                        <ul>
                            <li>Average temperatures</li>
                            <li>Regional variations</li>
                            <li>Temporal patterns</li>
                        </ul>
                    </li>
                    <li><strong>Units:</strong> Degrees Celsius</li>
                    <li><strong>Temporal Resolution:</strong> Annual averages</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        if st.checkbox("Show Temperature Dataset Sample"):
            st.dataframe(datasets['temp'].head())

    with dataset_tabs[4]:
        st.markdown("""
            <div class='text-area'>
                <h3>Rainfall Dataset</h3>
                <p>Precipitation data across regions:</p>
                <ul>
                    <li><strong>Measurements:</strong>
                        <ul>
                            <li>Annual rainfall amounts</li>
                            <li>Regional precipitation patterns</li>
                            <li>Seasonal variations</li>
                        </ul>
                    </li>
                    <li><strong>Units:</strong> Millimeters per year</li>
                    <li><strong>Coverage:</strong> Regional rainfall data</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        if st.checkbox("Show Rainfall Dataset Sample"):
            st.dataframe(datasets['rainfall'].head())

# Update show_homepage function to include dataset information
def show_homepage():
    """Display homepage with dataset information"""
    st.markdown("""
        <div class='centered-title'>
            <h1 style='color: #2C3E50; font-family: "Georgia", serif;'>🌾 CropInsight: Enhancing Agricultural Productivity</h1>
        </div>

        <div class='text-area'>
            <h2 style='text-align: center; color: #2980B9;'>Welcome to CropInsight</h2>
            <p style='text-align: center; font-size: 1.2rem;'>
            CropInsight aims to enhance agricultural productivity by providing tools that analyze how various factors affect crop yields.
            This platform is tailored for agricultural professionals, data scientists, policymakers, and researchers seeking to make informed decisions based on data-driven insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Add Understanding CropInsight section
    create_understanding_section()

    # Add dataset information
    show_dataset_information()        

def customer_analysis(datasets):
    """Customer Analysis Dashboard"""
    st.title("Customer Analysis Dashboard")

    # Create tabs for different analyses
    tabs = st.tabs([
        "Custom Visualization",
        "Correlation Analysis",
        "Geographic Analysis",
        "PCA Analysis",
        "ARIMA Forecast",
        "RBF-NN Interpolation"
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

def create_customer_visualization(df):
    """Create customizable visualizations for customers"""
    st.subheader("Scatter Plot")

    # Rename 'Item' to 'Crops'
    df = df.rename(columns={'Item': 'Crops'})

    col1, col2 = st.columns(2)
    with col1:
        x_axis = st.selectbox("Select X-axis", df.select_dtypes(include=[np.number]).columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis", df.select_dtypes(include=[np.number]).columns)

    color_by = st.selectbox("Color by", ['Crops', 'Area'])

    fig = px.scatter(df, x=x_axis, y=y_axis, color=df[color_by],
                     title=f"{y_axis} vs {x_axis} by {color_by}",
                     color_continuous_scale=px.colors.sequential.Viridis)  # Use a color scale without white background
    st.plotly_chart(fig, use_container_width=True)

def create_correlation_heatmap(df):
    """Create correlation heatmap for different crops with correlation coefficients displayed"""
    st.subheader("Correlation Analysis")

    # Rename 'Item' to 'Crops'
    df = df.rename(columns={'Item': 'Crops'})

    selected_crop = st.selectbox("Select Crop", ['All'] + list(df['Crops'].unique()))
    if selected_crop != 'All':
        crop_data = df[df['Crops'] == selected_crop]
    else:
        crop_data = df

    numeric_cols = crop_data.select_dtypes(include=[np.number]).columns
    corr_matrix = crop_data[numeric_cols].corr()

    # Use plotly.graph_objects to include annotations
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

def create_geographic_time_series(df):
    """Create geographic visualization with time slider"""
    st.subheader("Geographic Analysis")

    # Rename 'Item' to 'Crops'
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

def create_pca_analysis(df):
    """Create PCA visualization with option to select individual crops and separate vector plot"""
    st.subheader("PCA Analysis")

    # Rename 'Item' to 'Crops'
    df = df.rename(columns={'Item': 'Crops'})

    # User selects crop
    crop_options = ['All'] + list(df['Crops'].unique())
    selected_crop = st.selectbox("Select Crop for PCA", crop_options)

    if selected_crop != 'All':
        df_pca = df[df['Crops'] == selected_crop]
    else:
        df_pca = df

    numeric_cols = df_pca.select_dtypes(include=[np.number]).columns
    X = StandardScaler().fit_transform(df_pca[numeric_cols])

    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    # Remove vectors from the PCA plot
    fig = px.scatter(x=components[:, 0], y=components[:, 1],
                     color=df_pca['Area'],
                     title=f"PCA Analysis for {selected_crop}",
                     labels={'x': 'PC1 (Principal Component 1)', 'y': 'PC2 (Principal Component 2)'})
    st.plotly_chart(fig, use_container_width=True)

    # Add explanation for explained variance
    st.markdown(f"""
        <div class='text-area'>
            <h4>Explained Variance:</h4>
            <p>PC1 explains {pca.explained_variance_ratio_[0]:.2%} of the variance.</p>
            <p>PC2 explains {pca.explained_variance_ratio_[1]:.2%} of the variance.</p>
        </div>
    """, unsafe_allow_html=True)

def create_arima_forecast(df):
    """Create ARIMA forecast with training and testing data in different colors"""
    st.subheader("ARIMA Forecast")

    # Rename 'Item' to 'Crops'
    df = df.rename(columns={'Item': 'Crops'})

    selected_crop = st.selectbox("Select Crop for Forecast", df['Crops'].unique())
    p_value = st.slider("Select AR(p) Parameter", 1, 5, 2)
    test_size = st.slider("Select Test Data Size", 1, 10, 5)

    crop_data = df[df['Crops'] == selected_crop]['yield'].values

    # Split data into training and testing
    train_data = crop_data[:-test_size]
    test_data = crop_data[-test_size:]

    model = ARIMA(train_data, order=(p_value,1,1))
    results = model.fit()

    forecast = results.forecast(steps=test_size)

    # Plot training and testing data
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_data, name="Training Data", mode='lines'))
    fig.add_trace(go.Scatter(x=np.arange(len(train_data), len(train_data) + len(test_data)), y=test_data, name="Testing Data", mode='lines'))
    fig.update_layout(title=f"Training and Testing Data for {selected_crop}",
                      xaxis_title='Time',
                      yaxis_title='Yield')
    st.plotly_chart(fig, use_container_width=True)

    # Plot forecast vs testing data
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(y=test_data, name="Actual Testing Data", mode='lines'))
    fig_forecast.add_trace(go.Scatter(y=forecast, name="Forecast", mode='lines'))
    fig_forecast.update_layout(title=f"Forecast vs Actual Testing Data for {selected_crop}",
                               xaxis_title='Time',
                               yaxis_title='Yield')
    st.plotly_chart(fig_forecast, use_container_width=True)

def create_rbf_interpolation(df):
    """Create RBF interpolation with adjustable parameters and both 2D and 3D plots"""
    st.subheader("RBF-NN Interpolation")

    # Rename 'Item' to 'Crops'
    df = df.rename(columns={'Item': 'Crops'})

    selected_crop = st.selectbox("Select Crop for Interpolation", df['Crops'].unique())
    rbf_function = st.selectbox("Select RBF Function",
                               ['thin_plate_spline', 'multiquadric', 'gaussian'],
                               index=0)
    epsilon = st.slider("Select epsilon parameter", 0.1, 10.0, 1.0, 0.1)

    crop_data = df[df['Crops'] == selected_crop]
    X = crop_data[['year', 'average_rain_fall_mm_per_year']].values
    y = crop_data['yield'].values

    # Create and fit RBF interpolator
    rbf = RBFInterpolator(X, y,
                          kernel=rbf_function,
                          epsilon=epsilon)

    # Create grid for prediction
    xi = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    yi = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    xi, yi = np.meshgrid(xi, yi)
    grid_points = np.vstack([xi.ravel(), yi.ravel()]).T

    # Perform interpolation
    zi = rbf(grid_points).reshape(xi.shape)

    # Create 3D surface plot
    fig_3d = go.Figure(data=[
        go.Surface(x=xi, y=yi, z=zi)
    ])

    fig_3d.update_layout(
        title=f"3D RBF Interpolation for {selected_crop}",
        scene=dict(
            xaxis_title="Year",
            yaxis_title="Rainfall (mm)",
            zaxis_title="Yield"
        ),
        width=800,
        height=800
    )

    st.plotly_chart(fig_3d, use_container_width=True)

    # Create 2D contour plot
    fig_2d = go.Figure(data=
        go.Contour(
            x=xi[0],
            y=yi[:,0],
            z=zi,
            contours_coloring='heatmap',
            colorbar=dict(title='Yield')
        )
    )
    fig_2d.update_layout(
        title=f"2D RBF Interpolation for {selected_crop}",
        xaxis_title="Year",
        yaxis_title="Rainfall (mm)"
    )
    st.plotly_chart(fig_2d, use_container_width=True)

    # Add explanation of parameters
    st.markdown("""
        <div class='text-area'>
            <h4>Parameters Explanation:</h4>
            <ul>
                <li><strong>RBF Function:</strong> The type of radial basis function used for interpolation</li>
                <li><strong>Epsilon:</strong> Shape parameter that affects how local or global the interpolation is</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def technical_details(datasets):
    """Technical Analysis Dashboard"""
    st.title("Technical Analysis Dashboard")

    # Create tabs for different technical analyses
    tech_tabs = st.tabs([
        "Dataset Visualization",
        "Missing Values Analysis",
        "MICE Comparison"
    ])

    with tech_tabs[0]:
        create_technical_visualization(datasets)

    with tech_tabs[1]:
        visualize_missing_values(datasets)

    with tech_tabs[2]:
        compare_mice_results(datasets)

def create_technical_visualization(datasets):
    """Create technical visualizations for different datasets with color option"""
    st.subheader("Dataset Visualization")

    selected_dataset = st.selectbox("Select Dataset", list(datasets.keys()))
    df = datasets[selected_dataset]

    # Apply column drops for 'yield' and 'pesticides' datasets if not already applied
    if selected_dataset == 'yield':
        df = df.drop(columns=["Domain Code", "Domain", "Area Code", "Element Code",
                              "Item Code", "Element", "Year Code", "Unit"], errors='ignore')
        df.rename(columns={'Year': 'year', 'Item': 'Crops'}, inplace=True)
    elif selected_dataset == 'pesticides':
        df = df.drop(columns=["Domain", "Item", "Element", "Unit"], errors='ignore')
        df.rename(columns={'Year': 'year'}, inplace=True)
    else:
        df.rename(columns={'Year': 'year'}, inplace=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("Select X-axis", df.columns)
    with col2:
        y_axis = st.selectbox("Select Y-axis", df.columns)
    with col3:
        color_by = st.selectbox("Color by", [None] + list(df.columns))

    fig = px.scatter(df, x=x_axis, y=y_axis, color=df[color_by] if color_by else None,
                     title=f"{selected_dataset}: {y_axis} vs {x_axis}")
    st.plotly_chart(fig, use_container_width=True)

def visualize_missing_values(datasets):
    """Create missing values visualization using provided format"""
    st.subheader("Missing Values Analysis")

    # Plot Missing Value Heatmaps
    st.markdown("### Missing Values Distribution in Yield Data")
    plot_missing_heatmap(datasets['yield'], 'Value', 'Missing Values Distribution in Yield Data')

    st.markdown("### Missing Values Distribution in Pesticide Data")
    plot_missing_heatmap(datasets['pesticides'], 'Value', 'Missing Values Distribution in Pesticide Data')

    st.markdown("### Missing Values Distribution in Temperature Data")
    plot_missing_heatmap(datasets['temp'], 'avg_temp', 'Missing Values Distribution in Temperature Data')

    st.markdown("### Missing Values Distribution in Rainfall Data")
    plot_missing_heatmap(datasets['rainfall'], 'average_rain_fall_mm_per_year', 'Missing Values Distribution in Rainfall Data')

    # Add explanation
    st.markdown("""
        <div class='text-area'>
            <h4>Missing Values Handling Strategy:</h4>
            <p>
            The heatmaps above show the distribution of missing values across different areas and years.
            </p>
        </div>
    """, unsafe_allow_html=True)

def compare_mice_results(datasets):
    """Compare results before and after MICE imputation"""
    st.subheader("MICE Imputation Comparison")

    # Assuming we have original and MICE-imputed datasets
    dataset_type = st.selectbox("Select Dataset Type", ["yield", "pesticides"])
    df_original = datasets.get(dataset_type)
    df_imputed = datasets.get(f"{dataset_type}_imputed")

    if df_original is None or df_imputed is None:
        st.error("Cannot find imputed results for the selected dataset.")
        return

    # Apply column drops if necessary
    if dataset_type == 'yield':
        df_original = df_original.drop(columns=["Domain Code", "Domain", "Area Code", "Element Code",
                                                "Item Code", "Element", "Year Code", "Unit"], errors='ignore')
        df_imputed = df_imputed.drop(columns=["Domain Code", "Domain", "Area Code", "Element Code",
                                              "Item Code", "Element", "Year Code", "Unit"], errors='ignore')
        df_original.rename(columns={'Year': 'year', 'Item': 'Crops'}, inplace=True)
        df_imputed.rename(columns={'Year': 'year', 'Item': 'Crops'}, inplace=True)
    elif dataset_type == 'pesticides':
        df_original = df_original.drop(columns=["Domain", "Item", "Element", "Unit"], errors='ignore')
        df_imputed = df_imputed.drop(columns=["Domain", "Item", "Element", "Unit"], errors='ignore')
        df_original.rename(columns={'Year': 'year'}, inplace=True)
        df_imputed.rename(columns={'Year': 'year'}, inplace=True)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Original Data Distribution")
        fig1 = px.box(df_original, title="Original Data")
        st.plotly_chart(fig1)

    with col2:
        st.write("Imputed Data Distribution")
        fig2 = px.box(df_imputed, title="Imputed Data")
        st.plotly_chart(fig2)

# New Functions Provided by User
def plot_missing_heatmap(df, value_column, title, agg_method='first'):
    """
    Create a heatmap visualization of missing values across areas and years.

    Parameters:
    - df: Input DataFrame
    - value_column: Column containing the values to analyze
    - title: Title of the heatmap
    - agg_method: Method to aggregate duplicate entries (default: 'first')
    """
    # Validate input
    required_columns = ['Area', 'year', value_column]
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Required column '{col}' not found in the DataFrame")
            return

    # Select relevant columns
    df_subset = df[required_columns].copy()

    # Aggregate duplicates
    df_subset = df_subset.groupby(['Area', 'year'])[value_column].agg(agg_method).reset_index()

    # Create pivot table
    pivot = df_subset.pivot(index='year', columns='Area', values=value_column)

    # Create the plot
    plt.figure(figsize=(20, 10))

    # Generate heatmap of missing values
    sns.heatmap(pivot.isnull(),
                cbar=False,  # No color bar
                cmap='plasma',  # Color scheme
                xticklabels=True,
                yticklabels=True)

    # Customize plot
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('Area', fontsize=12)
    plt.ylabel('Year', fontsize=12)

    # Rotate x-axis labels for readability
    plt.xticks(rotation=90, ha='right')

    # Adjust y-axis ticks to prevent overcrowding
    num_years = pivot.shape[0]
    y_tick_locations = np.linspace(0, num_years-1, min(20, num_years)).astype(int)
    y_tick_labels = pivot.index[y_tick_locations]
    plt.yticks(y_tick_locations, y_tick_labels)

    # Ensure layout is tight
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(plt.gcf())
    plt.clf()  # Clear the figure after displaying

# Diagnostic Function to Check Duplicates
def diagnose_duplicates(df, columns):
    """
    Identify and print details about duplicate entries in specified columns.

    Parameters:
    - df: Input DataFrame
    - columns: List of columns to check for duplicates
    """
    duplicates = df[df.duplicated(subset=columns, keep=False)]

    st.write(f"Total duplicate entries: {len(duplicates)}")
    st.write("\nDuplicate Entries:")
    st.write(duplicates)

    # Groupby to see duplicate patterns
    duplicate_groups = duplicates.groupby(columns).size().reset_index(name='count')
    st.write("\nDuplicate Groups:")
    st.write(duplicate_groups[duplicate_groups['count'] > 1])

def main():
    """Main application function"""
    # Load data
    datasets = load_data()

    # Load and process background image
    img_str = load_and_process_bg_image()

    # Apply styling
    apply_styling(img_str)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Select Section", ["Home", "Customer Analysis", "Technical Details"])

    if section == "Home":
        show_homepage()
    elif section == "Customer Analysis":
        customer_analysis(datasets)
    elif section == "Technical Details":
        technical_details(datasets)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("An error occurred.")
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the input data and try again.")