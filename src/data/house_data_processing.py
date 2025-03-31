import re
import sys
import pandas as pd

from typing import Any
from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataCleaningConfig

class HouseDataStrategy(ABC):
    """
    Abstract base class that defines a strategy for handling house data.
    Subclasses must implement the `handle_data` method.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the given DataFrame and returns the cleaned DataFrame.
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
        """
        try:
            logging.info("Starting house data Pre-Processing...")
            
            df = data
            
            logging.info("Removing duplicate rows...")
            df.drop_duplicates(inplace=True)

            logging.info("Dropping unwanted columns...")
            df.drop(columns=['link', 'property_id'], inplace=True, errors='ignore')

            logging.info("Renaming columns...")
            df.rename(columns={'rate': 'price_per_sqft'}, inplace=True)

            logging.info("Cleaning 'society' column...")
            df['society'] = df['society'].astype(str).apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', name).strip().lower())
            df['society'] = df['society'].replace('nan', 'independent')

            logging.info("Processing price column...")
            df['price'] = df['price'].str.split(' ').apply(self.treat_price)

            logging.info("Cleaning 'price_per_sqft' column...")
            df['price_per_sqft'] = (
                df['price_per_sqft']
                .str.split('/')
                .str.get(0)
                .str.replace('₹', '')
                .str.replace(',', '')
                .str.strip()
                .astype(float)
            )

            logging.info("Cleaning 'bedRoom' column...")
            df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype(int)

            logging.info("Cleaning 'bathroom' column...")
            df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype(int)

            logging.info("Cleaning 'balcony' column...")
            df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

            logging.info("Handling missing values in 'additionalRoom' column...")
            df['additionalRoom'].fillna('not available', inplace=True)
            df['additionalRoom'] = df['additionalRoom'].str.lower()

            logging.info("Processing 'floorNum' column...")
            df['noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)
            df.rename(columns={'noOfFloor': 'floorNum'}, inplace=True)

            logging.info("Handling missing values in 'facing' column...")
            df['facing'].fillna('NA', inplace=True)

            logging.info("Calculating area based on price and price per square foot...")
            df['area'] = round((df['price'] * 10000000) / df['price_per_sqft'])

            logging.info("Adding 'property_type' column...")
            df.insert(loc=1, column='property_type', value='house')

            logging.info("House data pre-processing completed successfully.")
            
            return df
        
        except Exception as e:
            logging.error("Error occurred in cleaning house data", exc_info=True)
            raise MyException(e, sys)

class HouseDataCleaning(HouseDataStrategy):
    """
    Class responsible for applying a specified data cleaning strategy to the dataset.
    """
    def __init__(self, data: pd.DataFrame, strategy: HouseDataStrategy, config: DataCleaningConfig) -> None:
        """
        Initializes the HouseDataCleaning class with a dataset and a strategy.
        """
        self.config = config
        self.df = data
        self.strategy = strategy
        
    def handle_data(self) -> pd.DataFrame:
        """
        Executes the selected data cleaning strategy on the dataset.
        """
        logging.info("Starting data cleaning process...")
        cleaned_data = self.strategy.handle_data(self.df)
        logging.info("Data cleaning process completed successfully.")
        return cleaned_data
