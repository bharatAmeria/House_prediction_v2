import re
import sys
import pandas as pd

from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataCleaningConfig

class FlatsDataStrategy(ABC):
    """
    Abstract Class defining strategy for handling flats data.
    This class enforces the implementation of a `handle_data` method in any subclass,
    ensuring a standardized approach to processing flat-related datasets.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class FlatsDataPreProcessingStrategy(FlatsDataStrategy):
    """
    Data preprocessing strategy for cleaning and transforming flat-related datasets.
    This class provides various methods to preprocess and clean the dataset
    before further analysis or storage.
    """
    def treat_price(self, x):
        """
        Converts price values into float format.
        Handles cases where the price is in Lakhs and converts it to a standard float format.

        Args:
            x (list): A list where the first element is the price and the second element is the unit (if present).
        
        Returns:
            float: Processed price value.
        """
        if isinstance(x, float):  # Use isinstance for type checking
            return x
        if x[1] == 'Lac':
            return round(float(x[0]) / 100, 2)
        return round(float(x[0]), 2)

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and preprocesses the flats dataset.
        This includes renaming columns, handling missing values, and cleaning text fields.
        
        Args:
            data (pd.DataFrame): The input DataFrame containing raw flat data.
        
        Returns:
            pd.DataFrame: The cleaned and preprocessed dataset.
        """
        try:
            logging.info("Starting flats data Pre-Processing...")

            data = data.copy()

            # Dropping unnecessary columns
            data.drop(columns=['link', 'property_id'], inplace=True)
            
            # Renaming columns for clarity
            data.rename(columns={'area': 'price_per_sqft'}, inplace=True)
            
            # Cleaning 'society' column by removing rating stars and converting to lowercase
            data['society'] = data['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()

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

            # Cleaning 'bedRoom' column and converting to integer
            data = data[~data['bedRoom'].isnull()]
            data['bedRoom'] = data['bedRoom'].str.split(' ').str.get(0).astype(int)

            # Cleaning 'bathroom' column
            data['bathroom'] = data['bathroom'].str.split(' ').str.get(0).astype(int)

            # Cleaning 'balcony' column and replacing 'No' with '0'
            data['balcony'] = data['balcony'].str.split(' ').str.get(0).str.replace('No', '0')

            # Handling missing values for 'additionalRoom'
            data['additionalRoom'].fillna('not available', inplace=True)
            data['additionalRoom'] = data['additionalRoom'].str.lower()

            # Processing 'floorNum' column
            data['floorNum'] = (
                data['floorNum']
                .str.split(' ')
                .str.get(0)
                .replace('Ground', '0')
                .str.replace('Basement', '-1')
                .str.replace('Lower', '0')
                .str.extract(r'(\d+)')
            )

            # Handling missing values for 'facing'
            data['facing'].fillna('NA', inplace=True)

            # Calculating area using price and price per square foot
            data.insert(loc=4, column='area', value=round((data['price'] * 10000000) / data['price_per_sqft']))

            # Adding property_type column
            data.insert(loc=1, column='property_type', value='flat')

            logging.info("Flats data pre-processing completed successfully.\n")
            return data

        except Exception as e:
            logging.error("Error occurred in cleaning flats data", exc_info=True)
            raise MyException(e, sys)

class FlatsDataCleaning(FlatsDataStrategy):
    """
    Data cleaning class that applies a specified data cleaning strategy.
    This class allows the implementation of different cleaning strategies for flats data.
    """
    def __init__(self, data: pd.DataFrame, strategy: FlatsDataStrategy, config: DataCleaningConfig) -> None:
        """
        Initializes the DataCleaning class with a specific strategy.

        Args:
            data (pd.DataFrame): The input DataFrame containing raw flat data.
            strategy (FlatsDataStrategy): The data cleaning strategy to be applied.
            config (DataCleaningConfig): Configuration object containing cleaning parameters.
        """
        self.config = config
        self.df = data
        self.strategy = strategy
        
    def handle_data(self) -> pd.DataFrame:
        """
        Handles data using the provided cleaning strategy.
        Logs the start and completion of the data cleaning process.

        Returns:
            pd.DataFrame: The cleaned dataset after applying the strategy.
        """
        logging.info("Starting data cleaning process...")
        cleaned_data = self.strategy.handle_data(self.df)
        logging.info("Data cleaning process completed successfully.")
        return cleaned_data
