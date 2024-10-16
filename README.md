# CMSE-830

# Agricultural Yield Prediction

## Overview
The ultimate goal of the project is to predict crop yields by analyzing and merging two datasets containing various agricultural-related features. These datasets can be used in subsequent machine learning to help researchers and farmers optimize agricultural production and related resource management.

## Datasets
1. **Dataset 1**: 
   - Contains **1,000,000 samples** with features including:
     - **Region**: The geographical region where the crop is grown (North, East, South, West).
     - **Soil_Type**: The type of soil in which the crop is planted (Clay, Sandy, Loam, Silt, Peaty, Chalky).
     - **Crop**: The type of crop grown (Wheat, Rice, Maize, Barley, Soybean, Cotton).
     - **Rainfall_mm**: The amount of rainfall received in millimeters during the crop growth period.
     - **Temperature_Celsius**: The average temperature during the crop growth period, measured in degrees Celsius.
     - **Fertilizer_Used**: Indicates whether fertilizer was applied (True = Yes, False = No).
     - **Irrigation_Used**: Indicates whether irrigation was used during the crop growth period (True = Yes, False = No).
     - **Weather_Condition**: The predominant weather condition during the growing season (Sunny, Rainy, Cloudy).
     - **Days_to_Harvest**: The number of days taken for the crop to be harvested after planting.
     - **Yield_tons_per_hectare**: The total crop yield produced, measured in tons per hectare.

2. **Dataset 2**: 
   - Contains **20,000 entries** with features including:
     - **Soil_Quality**: Soil quality index, ranging from 50 to 100.
     - **Seed_Variety**: Binary indicator of seed variety, where 1 represents a high-yield variety.
     - **Fertilizer_Amount_kg_per_hectare**: The amount of fertilizer used in kilograms per hectare.
     - **Sunny_Days**: The number of sunny days during the growing season.
     - **Rainfall_mm**: Total rainfall received during the growing season in millimeters.
     - **Irrigation_Schedule**: The number of irrigations during the growing season.
     - **Yield_kg_per_hectare**: The agricultural yield in kilograms per hectare.

