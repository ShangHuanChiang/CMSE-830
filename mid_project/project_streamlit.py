import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Define URLs for the crop datasets from your GitHub repository
DATASET_URLS = {
    "Yield Dataset_2": "https://github.com/ShangHuanChiang/CMSE-830/raw/main/mid_project/agricultural_yield.csv",
    "Wheat": "https://github.com/ShangHuanChiang/CMSE-830/raw/main/mid_project/Wheat.csv",
    "Rice": "https://github.com/ShangHuanChiang/CMSE-830/raw/main/mid_project/Rice.csv",
    "Maize": "https://github.com/ShangHuanChiang/CMSE-830/raw/main/mid_project/Maize.csv",
    "Barley": "https://github.com/ShangHuanChiang/CMSE-830/raw/main/mid_project/Barley.csv",
    "Soybean": "https://github.com/ShangHuanChiang/CMSE-830/raw/main/mid_project/Soybean.csv",
    "Cotton": "https://github.com/ShangHuanChiang/CMSE-830/raw/main/mid_project/Cotton.csv"
}

# Function to load data from the selected URL with error handling
def load_data(url):
    try:
        data = pd.read_csv(url, sep=',', on_bad_lines='skip')
        return data
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return None

# PCA function
def apply_pca_and_plot(data, target_column, dataset_name, n_components=2):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(data=X_pca, columns=[f'Principal Component {i+1}' for i in range(n_components)])
    pca_df['Yield'] = y.values

    explained_variance = pca.explained_variance_ratio_

    fig = px.scatter(pca_df,
                     x='Principal Component 1',
                     y='Principal Component 2',
                     color='Yield',
                     title=f'Interactive PCA of {dataset_name} Yield Data',
                     labels={'Principal Component 1': f'Principal Component 1<br>(Explained Variance: {explained_variance[0]:.2%})',
                             'Principal Component 2': f'Principal Component 2<br>(Explained Variance: {explained_variance[1]:.2%})'},
                     color_continuous_scale=px.colors.sequential.Viridis,
                     hover_data=['Yield']
                    )

    fig.update_layout(width=1000, height=600)
    st.plotly_chart(fig)

# K-Means Clustering function
def apply_kmeans_and_plot(data, target_column):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.drop(columns=[target_column]))

    # Elbow Method for Optimal k
    K = range(1, 15)
    cost = []
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=99, n_init='auto')
        kmeans.fit(scaled_data)
        cost.append(kmeans.inertia_)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(K, cost, 'o-')
    plt.xlabel('k')
    plt.ylabel('Cost')
    plt.title('Elbow Method For Optimal k')

    kmeans_optimal = KMeans(n_clusters=3, random_state=99, n_init='auto')
    kmeans_optimal.fit(scaled_data)

    km_labels = pd.DataFrame(kmeans_optimal.labels_, columns=['Cluster'])
    clustered_data = pd.concat([data.reset_index(drop=True), km_labels], axis=1)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x=clustered_data['Rainfall_mm'], y=clustered_data[target_column], hue=clustered_data['Cluster'], palette='Set1', alpha=0.6)
    plt.title('K-means Clustering Results')

    plt.tight_layout()
    st.pyplot(plt)

# Streamlit app layout
st.title("Agricultural Yield Data Analysis and Visualization - Crop Dataset")

analysis_type = st.sidebar.selectbox("Select Analysis Type", 
                                     ["Dataset Preview", "Correlation Heatmap Analysis", "Clustering Analysis"])

# Dataset Preview Section
if analysis_type == "Dataset Preview":
    dataset_type = st.sidebar.selectbox("Dataset Preview", options=["Crop Subsets", "Yield Dataset_2"])

    
    if dataset_type == "Crop Subsets":
        st.subheader("Crop Subset Information")
        selected_subset = st.sidebar.selectbox("Choose a Crop Subset", options=list(DATASET_URLS.keys())[1:])

        crop_subset_data = load_data(DATASET_URLS[selected_subset])
        if crop_subset_data is not None:
            st.write(f"#### {selected_subset} Dataset")
            st.write(crop_subset_data.head(20))
            st.write("#### Basic Statistical Information")
            st.write(crop_subset_data.describe())
        
            # Define features for Crop Subsets
            numeric_features = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest', 'Yield_tons_per_hectare']
            categorical_features = {
                'Region': ['Region_East', 'Region_North', 'Region_South', 'Region_West'],
                'Soil_Type': ['Soil_Type_Chalky', 'Soil_Type_Clay', 'Soil_Type_Loam', 'Soil_Type_Peaty', 'Soil_Type_Sandy', 'Soil_Type_Silt'],
                'Fertilizer_Used': ['Fertilizer_Used_False', 'Fertilizer_Used_True'],
                'Irrigation_Used': ['Irrigation_Used_False', 'Irrigation_Used_True'],
                'Weather_Condition': ['Weather_Condition_Cloudy', 'Weather_Condition_Rainy', 'Weather_Condition_Sunny']
            }

            # Combine numeric features and categorical features for selection
            all_features = numeric_features + list(categorical_features.keys())
            selected_feature = st.selectbox("Select a feature to display:", all_features, key="feature_selection")

            # Plot based on the selected feature
            plt.figure(figsize=(10, 6))
            
            if selected_feature in categorical_features:
                # Count occurrences of sub-features for categorical features
                sub_features = categorical_features[selected_feature]
                count_data = {sub_feature: crop_subset_data[sub_feature].sum() for sub_feature in sub_features}
                combined_data = pd.DataFrame(list(count_data.items()), columns=['Sub_Feature', 'Count'])

                sns.barplot(data=combined_data, x='Sub_Feature', y='Count')
                plt.title(f'{selected_feature} Counts')
                plt.xlabel('Features')
                plt.ylabel('Counts')

            elif selected_feature in numeric_features:
                sns.histplot(data=crop_subset_data, x=selected_feature, kde=True)
                plt.title(f'{selected_feature} Distribution')
                plt.xlabel(selected_feature)
                plt.ylabel('Counts')

            st.pyplot(plt)
    elif dataset_type == "Yield Dataset_2":
        yield_data = load_data(DATASET_URLS["Yield Dataset_2"])
        if yield_data is not None:
            st.write("#### Yield Dataset Information")
            st.write(yield_data.head(20))
            st.write("#### Basic Statistical Information")
            st.write(yield_data.describe())
        
            # Feature selection for histogram
            selected_feature = st.selectbox("Select a feature for histogram", yield_data.columns)

            # Plot histogram for the selected feature
            plt.figure(figsize=(10, 6))
            sns.histplot(data=yield_data, x=selected_feature, kde=True)
            plt.title(f'{selected_feature} Distribution')
            plt.xlabel(selected_feature)
            plt.ylabel('Counts')
            
            st.pyplot(plt)

# Correlation Heatmap Analysis Section
if analysis_type == "Correlation Heatmap Analysis":
    st.subheader("Correlation Heatmap Analysis")

    dataset_type = st.sidebar.selectbox("Select Dataset", ["Crop Subsets", "Yield Dataset_2"])

    if dataset_type == "Crop Subsets":
        selected_subset = st.sidebar.selectbox("Choose a Crop Subset", options=list(DATASET_URLS.keys())[1:])

        one_hot_encoded_crop_data = load_data(DATASET_URLS[selected_subset])
        if one_hot_encoded_crop_data is not None:
            st.write(f"##### {selected_subset} Correlation Heatmap")
            correlation_matrix = one_hot_encoded_crop_data.corr().values
            feature_names = one_hot_encoded_crop_data.columns.tolist()

            correlation_text = [[f'{value:.2f}' for value in row] for row in correlation_matrix]

            fig = ff.create_annotated_heatmap(
                z=correlation_matrix,
                x=feature_names,
                y=feature_names,
                annotation_text=correlation_text,
                showscale=True
            )

            fig.update_layout(
                height=700,
                width=1400,
                title='',
                title_x=0.5
            )

            st.plotly_chart(fig)

    elif dataset_type == "Yield Dataset_2":
        st.write("##### Yield Dataset_2 Correlation Heatmap")

        one_hot_encoded_yield_data = load_data(DATASET_URLS["Yield Dataset_2"])
        if one_hot_encoded_yield_data is not None:
            correlation_matrix = one_hot_encoded_yield_data.corr().values
            feature_names = one_hot_encoded_yield_data.columns.tolist()

            correlation_text = [[f'{value:.2f}' for value in row] for row in correlation_matrix]

            fig = ff.create_annotated_heatmap(
                z=correlation_matrix,
                x=feature_names,
                y=feature_names,
                annotation_text=correlation_text,
                showscale=True
            )

            fig.update_layout(
                height=700,
                width=1400,
                title='',
                title_x=0.5
            )

            st.plotly_chart(fig)

# Clustering Analysis Section
if analysis_type == "Clustering Analysis":
    st.subheader("Clustering Analysis")

    dataset_type = st.sidebar.selectbox("Select Dataset", ["Crop Subsets", "Yield Dataset_2"])
    
    if dataset_type == "Crop Subsets":
        selected_subset = st.sidebar.selectbox("Choose a Crop Subset", options=list(DATASET_URLS.keys())[1:])

        crop_subset_data = load_data(DATASET_URLS[selected_subset])
        if crop_subset_data is not None:
            st.write("##### K-means clustering")
            apply_kmeans_and_plot(crop_subset_data, target_column='Yield_tons_per_hectare')
            apply_pca_and_plot(crop_subset_data, target_column='Yield_tons_per_hectare', dataset_name=selected_subset)

    elif dataset_type == "Yield Dataset_2":
        yield_data = load_data(DATASET_URLS["Yield Dataset_2"])
        if yield_data is not None:
            st.write("##### K-means clustering")
            apply_kmeans_and_plot(yield_data, target_column='Yield_kg_per_hectare')
            apply_pca_and_plot(yield_data, target_column='Yield_kg_per_hectare', dataset_name="Yield Dataset_2")
