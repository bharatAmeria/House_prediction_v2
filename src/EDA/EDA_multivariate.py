import re
import sys
import pandas as pd

from abc import ABC, abstractmethod
from typing import Union
from src.logger import logging
from src.exception import MyException


class EDA_multivariateStrategy(ABC):
    @abstractmethod
    def handle_EDA(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class EDA_Multivariate(EDA_multivariateStrategy):
    def handle_EDA(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            df = df[df['built_up_area'] != 737147]

            logging.info("Multivariate EDA started")
            # check outliers
            df[df['price_per_sqft'] > 100000][['property_type',
                                               'society',
                                               'sector',
                                               'price',
                                               'price_per_sqft',
                                               'area','areaWithType', 
                                               'super_built_up_area', 
                                               'built_up_area', 
                                               'carpet_area']]
            
            # checking outliers
            df[df['bedRoom'] >= 10]

            # checking for outliers
            df[(df['property_type'] == 'house') & (df['floorNum'] > 10)]

            # Group by 'sector' and calculate the average price
            avg_price_per_sector = df.groupby('sector')['price'].mean().reset_index()
            avg_price_per_sector['sector_number'] = avg_price_per_sector['sector'].apply(self.extract_sector_number)

            # Sort by sector number
            df = avg_price_per_sector.sort_values(by='sector_number')
            avg_price_per_sqft_sector = df.groupby('sector')['price_per_sqft'].mean().reset_index()

            avg_price_per_sqft_sector['sector_number'] = avg_price_per_sqft_sector['sector'].apply(self.extract_sector_number)

            # Sort by sector number
            df = avg_price_per_sqft_sector.sort_values(by='sector_number')

            luxury_score = df.groupby('sector')['luxury_score'].mean().reset_index()

            luxury_score['sector_number'] = luxury_score['sector'].apply(self.extract_sector_number)

            # Sort by sector number
            df = luxury_score.sort_values(by='sector_number')
            df.corr()['price'].sort_values(ascending=False)
            logging.info("Multivariate EDA started")

        except Exception as e:
            raise MyException(e, sys)

        
    # Function to extract sector numbers
    def extract_sector_number(self, sector_name):
        try:

            match = re.search(r'\d+', sector_name)
            if match:
                return int(match.group())
            else:
                return float('inf')  # Return a large number for non-numbered sectors

        except Exception as e:
            raise MyException(e, sys)
        
class EDAPreProcessing(EDA_multivariateStrategy):
    """
    Data cleaning class which preprocesses the data
    """
    def __init__(self, data: pd.DataFrame, strategy: EDA_multivariateStrategy) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.df = data
        self.strategy = strategy

    def handle_EDA(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_EDA(self.df)