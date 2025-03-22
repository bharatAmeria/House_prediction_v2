import re
import sys
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataCleaningConfig

class HouseDataStrategy(ABC):
    """
    Abstract Class defining strategy for handling house data.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DataPreprocessStrategy(HouseDataStrategy):
    """
    Data preprocessing strategy that preprocesses the house data.
    """
    def treat_price(self, x):
        """
        Converts price values into float format.
        """
        if isinstance(x, float):  # Use isinstance for type checking
            return x
        if x[1] == 'Lac':
            return round(float(x[0]) / 100, 2)
        return round(float(x[0]), 2)

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the house dataset.
        """
        try:
            logging.info("Starting house data Cleaning...")

            data = data.copy()
            # Removing duplicates
            data.drop_duplicates(inplace=True)

            # Dropping unwanted columns
            data.drop(columns=['link', 'property_id'], inplace=True, errors='ignore')

            # Renaming columns
            data.rename(columns={'rate': 'price_per_sqft'}, inplace=True)

            # Cleaning 'society' column
            data['society'] = data['society'].astype(str).apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', name).strip().lower())
            data['society'] = data['society'].replace('nan', 'independent')

            # Filtering out 'Price on Request' values
            data = data[data['price'] != 'Price on Request']

            # Processing price column
            data['price'] = data['price'].str.split(' ').apply(self.treat_price)

            # Processing 'price_per_sqft' column
            data['price_per_sqft'] = (
                data['price_per_sqft']
                .str.split('/')
                .str.get(0)
                .str.replace('₹', '')
                .str.replace(',', '')
                .str.strip()
                .astype(float)
            )

            # Cleaning 'bedRoom' column
            data = data[~data['bedRoom'].isnull()]
            data['bedRoom'] = data['bedRoom'].str.split(' ').str.get(0).astype(int)

            # Cleaning 'bathroom' column
            data['bathroom'] = data['bathroom'].str.split(' ').str.get(0).astype(int)

            # Cleaning 'balcony' column
            data['balcony'] = data['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

            # Handling missing values for 'additionalRoom'
            data['additionalRoom'].fillna('not available', inplace=True)
            data['additionalRoom'] = data['additionalRoom'].str.lower()

            # Processing 'floorNum' column
            data['noOfFloor'] = data['noOfFloor'].str.split(' ').str.get(0)
            data.rename(columns={'noOfFloor': 'floorNum'}, inplace=True)

            # Handling missing values for 'facing'
            data['facing'].fillna('NA', inplace=True)

            # Calculating area
            data['area'] = round((data['price'] * 10000000) / data['price_per_sqft'])

            # Adding property_type column
            data.insert(loc=1, column='property_type', value='house')

            logging.info("House data cleaning completed successfully.")

            return data

        except Exception as e:
            logging.error("Error occurred in cleaning house data", exc_info=True)
            raise MyException(e, sys)

class DataCleaning(HouseDataStrategy):
    """
    Data cleaning class that applies a specified data cleaning strategy.
    """
    def __init__(self, data: pd.DataFrame, strategy: HouseDataStrategy, config: DataCleaningConfig) -> None:
        """
        Initializes the DataCleaning class with a specific strategy.
        """
        self.config = config
        self.df = data
        self.strategy = strategy
        

    def handle_data(self) -> pd.DataFrame:
        """
        Handles data using the provided strategy.
        """
        logging.info("Starting data cleaning process...")
        cleaned_data = self.strategy.handle_data(self.df)
        logging.info("Data cleaning process completed successfully.")
        return cleaned_data

        