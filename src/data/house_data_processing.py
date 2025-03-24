import re
import sys
import pandas as pd

from typing import Any
from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataCleaningConfig

class DataCleaningConfig:
    """
    Configuration class for data cleaning operations.
    """
    pass

class HouseDataStrategy(ABC):
    """
    Abstract base class that defines a strategy for handling house data.
    Subclasses must implement the `handle_data` method.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the given DataFrame and returns the cleaned DataFrame.
        
        Parameters:
        data (pd.DataFrame): The input house dataset to be processed.

        Returns:
        pd.DataFrame: The processed dataset after applying cleaning strategies.
        """
        pass

class HouseDataPreProcessingStrategy(HouseDataStrategy):
    """
    Concrete strategy class that performs preprocessing on the house dataset.
    This includes data cleaning, formatting, and feature engineering.
    """
    def treat_price(self, x: Any) -> float:
        """
        Converts price values into a float format based on unit conventions.
        
        Parameters:
        x (Any): The price value in different formats (e.g., string, float).
        
        Returns:
        float: The processed price value in float format.
        """
        if isinstance(x, float):  # Ensure correct type handling
            return x
        if x[1] == 'Lac':
            return round(float(x[0]) / 100, 2)
        return round(float(x[0]), 2)

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the house dataset by handling missing values,
        formatting columns, and extracting relevant information.
        
        Parameters:
        data (pd.DataFrame): The input raw dataset containing house details.
        
        Returns:
        pd.DataFrame: The cleaned and preprocessed house dataset.
        """
        try:
            logging.info("Starting house data Pre-Processing...")
            
            data = data.copy()
            # Removing duplicate rows
            data.drop_duplicates(inplace=True)

            # Dropping unwanted columns
            data.drop(columns=['link', 'property_id'], inplace=True, errors='ignore')

            # Renaming columns for clarity
            data.rename(columns={'rate': 'price_per_sqft'}, inplace=True)

            # Cleaning 'society' column by removing unwanted characters
            data['society'] = data['society'].astype(str).apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', name).strip().lower())
            data['society'] = data['society'].replace('nan', 'independent')

            # Filtering out 'Price on Request' values
            data = data[data['price'] != 'Price on Request']

            # Processing price column by converting it into float
            data['price'] = data['price'].str.split(' ').apply(self.treat_price)

            # Cleaning 'price_per_sqft' column by removing currency symbols and commas
            data['price_per_sqft'] = (
                data['price_per_sqft']
                .str.split('/')
                .str.get(0)
                .str.replace('₹', '')
                .str.replace(',', '')
                .str.strip()
                .astype(float)
            )

            # Cleaning 'bedRoom' column and converting it to integer
            data = data[~data['bedRoom'].isnull()]
            data['bedRoom'] = data['bedRoom'].str.split(' ').str.get(0).astype(int)

            # Cleaning 'bathroom' column and converting it to integer
            data['bathroom'] = data['bathroom'].str.split(' ').str.get(0).astype(int)

            # Cleaning 'balcony' column and handling missing values
            data['balcony'] = data['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

            # Handling missing values in 'additionalRoom' column
            data['additionalRoom'].fillna('not available', inplace=True)
            data['additionalRoom'] = data['additionalRoom'].str.lower()

            # Processing 'floorNum' column
            data['noOfFloor'] = data['noOfFloor'].str.split(' ').str.get(0)
            data.rename(columns={'noOfFloor': 'floorNum'}, inplace=True)

            # Handling missing values in 'facing' column
            data['facing'].fillna('NA', inplace=True)

            # Calculating the area based on price and price per square foot
            data['area'] = round((data['price'] * 10000000) / data['price_per_sqft'])

            # Adding a new 'property_type' column
            data.insert(loc=1, column='property_type', value='house')

            logging.info("House data pre-processing completed successfully.\n")
            
            return data
        
        except Exception as e:
            logging.error("Error occurred in cleaning house data", exc_info=True)
            raise MyException(e, sys)

class HouseDataCleaning(HouseDataStrategy):
    """
    Class responsible for applying a specified data cleaning strategy to the dataset.
    It provides a flexible approach by allowing different strategies to be used.
    """
    def __init__(self, data: pd.DataFrame, strategy: HouseDataStrategy, config: DataCleaningConfig) -> None:
        """
        Initializes the HouseDataCleaning class with a dataset and a strategy.
        
        Parameters:
        data (pd.DataFrame): The dataset that needs to be cleaned.
        strategy (HouseDataStrategy): The strategy to be used for data cleaning.
        config (DataCleaningConfig): Configuration settings for cleaning operations.
        """
        self.config = config
        self.df = data
        self.strategy = strategy
        
    def handle_data(self) -> pd.DataFrame:
        """
        Executes the selected data cleaning strategy on the dataset.
        
        Returns:
        pd.DataFrame: The cleaned dataset after applying the strategy.
        """
        logging.info("Starting data cleaning process...")
        cleaned_data = self.strategy.handle_data(self.df)
        logging.info("Data cleaning process completed successfully.")
        return cleaned_data
