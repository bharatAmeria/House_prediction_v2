import ast
import re
from typing import Union

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from src.constants import RANDOM_STATE, WCSS_REDUCED, N_CLUSTERS, WEIGHTS
from src.logger import logging
from src.exception import MyException
from src.config.configuration import ConfigurationManager


class FeatureEngineeringStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_FE(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class FeatureEngineeringConfig(FeatureEngineeringStrategy):


    def handle_FE(self, data: pd.DataFrame, config=ConfigurationManager()) -> pd.DataFrame:
        """
        Feature engineering strategy which preprocesses the data.
        """
        try:
            logging.info("Starting feature engineering process.")
            config_ = config.get_data_cleaning_config()
            df = data

            logging.info("Extracting and converting Super Built-up area.")
            df['super_built_up_area'] = df['areaWithType'].apply(self.get_super_built_up_area)
            df['super_built_up_area'] = df.apply(
                lambda x: self.convert_to_sqft(x['areaWithType'], x['super_built_up_area']), axis=1)

            logging.info("Extracting and converting Built-Up area.")
            df['built_up_area'] = df['areaWithType'].apply(lambda x: self.get_area(x, 'Built Up area'))
            df['built_up_area'] = df.apply(lambda x: self.convert_to_sqft(x['areaWithType'], x['built_up_area']), axis=1)

            logging.info("Extracting and converting Carpet area.")
            df['carpet_area'] = df['areaWithType'].apply(lambda x: self.get_area(x, 'Carpet area'))
            df['carpet_area'] = df.apply(lambda x: self.convert_to_sqft(x['areaWithType'], x['carpet_area']), axis=1)

            logging.info("Handling missing values in area-related fields.")
            all_nan_df = df[((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))][['price','property_type','area','areaWithType','super_built_up_area','built_up_area','carpet_area']]
            all_nan_df['built_up_area'] = all_nan_df['areaWithType'].apply(self.extract_plot_area)
            all_nan_df['built_up_area'] = all_nan_df.apply(self.convert_scale, axis=1)
            # update the original dataframe
            df.update(all_nan_df)

            logging.info("Extracting additional room details.")
            new_cols = ['study room', 'servant room', 'store room', 'pooja room', 'others']
            for col in new_cols:
                df[col] = df['additionalRoom'].str.contains(col).astype(int)

            logging.info("Categorizing property age possession.")
            df['agePossession'] = df['agePossession'].apply(self.categorize_age_possession)

            logging.info("Processing furnishing details.")

            all_furnishings = []
            for detail in df['furnishDetails'].dropna():
                furnishings = detail.replace('[', '').replace(']', '').replace("'", "").split(', ')
                all_furnishings.extend(furnishings)
            unique_furnishings = list(set(all_furnishings))

            # Simplify the furnishings list by removing "No" prefix and numbers
            columns_to_include = [re.sub(r'No |\d+', '', furnishing).strip() for furnishing in unique_furnishings]
            columns_to_include = list(set(columns_to_include))  # Get unique furnishings
            columns_to_include = [furnishing for furnishing in columns_to_include if furnishing]  # Remove empty strings

            # Create new columns for each unique furnishing and populate with counts
            for furnishing in columns_to_include:
                df[furnishing] = df['furnishDetails'].apply(lambda x: self.get_furnishing_count(x, furnishing))

            # Create the new dataframe with the required columns
            furnishings_df = df[['furnishDetails'] + columns_to_include]
            furnishings_df.drop(columns=['furnishDetails'],inplace=True)

            logging.info("Scaling furnishing details.")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(furnishings_df)

            logging.info("Applying KMeans clustering on furnishing details.")
            kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
            kmeans.fit(scaled_data)

            # Predict the cluster assignments for each row
            cluster_assignments = kmeans.predict(scaled_data)
            df = df.iloc[:,:-18]
            df['furnishing_type'] = cluster_assignments

            
            logging.info("Handling missing features data.")
            app_df = pd.read_csv(config_.gurgaon_appartments_data)
            app_df['PropertyName'] = app_df['PropertyName'].str.lower()
            temp_df = df[df['features'].isnull()]
            df.loc[temp_df.index, 'features'] = temp_df.merge(app_df, left_on='society', right_on='PropertyName', how='left')['TopFacilities'].values

            logging.info("Converting features into binary format.")
            df['features_list'] = df['features'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) and x.startswith('[') else [])
            mlb = MultiLabelBinarizer()
            features_binary_matrix = mlb.fit_transform(df['features_list'])
            features_binary_df = pd.DataFrame(features_binary_matrix, columns=mlb.classes_)

            logging.info("Calculating luxury score.")
            luxury_score = features_binary_df[list(WEIGHTS.keys())].multiply(list(WEIGHTS.values())).sum(axis=1)
            df['luxury_score'] = luxury_score

            logging.info("Dropping unnecessary columns.")
            df.drop(columns=['nearbyLocations', 'furnishDetails', 'features', 'features_list', 'additionalRoom'], inplace=True)
            df = df[df['built_up_area'] != 737147]

            logging.info("Computing sector-wise statistics.")
            df.to_csv(config_.cleaned_gurgaon_data, index=False)
            logging.info("Feature engineering process completed successfully.")
            return df
        except Exception as e:
            logging.error(f"Error in feature engineering: {e}")
            raise e
        
    # Function to extract sector numbers
    def extract_sector_number(self, sector_name):
        match = re.search(r'\d+', sector_name)
        if match:
            return int(match.group())
        else:
            return float('inf')  # Return a large number for non-numbered sectors

        
    def convert_scale(self, row):
                if np.isnan(row['area']) or np.isnan(row['built_up_area']):
                    return row['built_up_area']
                else:
                    if round(row['area'] / row['built_up_area']) == 9.0:
                        return row['built_up_area'] * 9
                    elif round(row['area'] / row['built_up_area']) == 11.0:
                        return row['built_up_area'] * 10.7
                    else:
                        return row['built_up_area']
        
    """ Age of Possession"""
    def categorize_age_possession(self, value):
        if pd.isna(value):
            return "Undefined"
        if "0 to 1 Year Old" in value or "Within 6 months" in value or "Within 3 months" in value:
            return "New Property"
        if "1 to 5 Year Old" in value:
            return "Relatively New"
        if "5 to 10 Year Old" in value:
            return "Moderately Old"
        if "10+ Year Old" in value:
            return "old property"
        if "Under Construction" in value or "By" in value:
            return "Under Construction"
        try:
            int(value.split(" ")[-1])
            return "Under Construction"
        except:
            return "Undefined"
        
    """ Furnishing Details """
    # Define a function to extract the count of a furnishing from the furnishDetails
    def get_furnishing_count(self, details, furnishing):
        if isinstance(details, str):
            if f"No {furnishing}" in details:
                return 0
            pattern = re.compile(f"(\d+) {furnishing}")
            match = pattern.search(details)
            if match:
                return int(match.group(1))
            elif furnishing in details:
                return 1
        return 0
    
    # Function to extract plot area from 'areaWithType' column
    def extract_plot_area(self, area_with_type):
        match = re.search(r'Plot area (\d+\.?\d*)', area_with_type)
        return float(match.group(1)) if match else None


    # This function extracts the Super Built-up area
    def get_super_built_up_area(self, text):
        match = re.search(r'Super Built up area (\d+\.?\d*)', text)
        if match:
            return float(match.group(1))
        return None

    # This function extracts the Built-Up area or Carpet area
    def get_area(self, text, area_type):
        match = re.search(area_type + r'\s*:\s*(\d+\.?\d*)', text)
        if match:
            return float(match.group(1))
        return None

    # This function checks if the area is provided in sq.m. and converts it to sqft if needed
    def convert_to_sqft(self, text, area_value):
        if area_value is None:
            return None
        match = re.search(r'{} \((\d+\.?\d*) sq.m.\)'.format(area_value), text)
        if match:
            sq_m_value = float(match.group(1))
            return sq_m_value * 10.7639  # conversion factor from sq.m. to sqft
        return area_value

class FeatureEngineering:
    """
    Feature engineering class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: FeatureEngineeringStrategy) -> None:
        self.df = data
        self.strategy = strategy

    def handle_FE(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_FE(self.df)