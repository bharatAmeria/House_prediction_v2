import sys
import pandas as pd

from typing import Union
from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException

class OutlierStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_outlier(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass
    
class OutlierProcessStrategy(OutlierStrategy):
    """
    Removes outlier columns which are not required, fills missing values with median average values.
    """ 
    def handle_outlier(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
             
            df = data

            outlier_price = self.detect_outliers_iqr(df, 'price')
            outlier_sqft = self.detect_outliers_iqr(df, 'area')

            outlier_price = outlier_price.sort_values('price', ascending=False).head(20)
            outlier_sqft['area'] = outlier_sqft['area'].apply(lambda x: x * 9 if x < 1000 else x)
            outlier_sqft['price_per_sqft'] = round((outlier_sqft['price'] * 10000000) / outlier_sqft['area'])

            df.update(outlier_sqft)
            df.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True, errors='ignore')
            
            df.loc[48, 'area'] = 115 * 9
            df.loc[300, 'area'] = 7250
            df.loc[2666, 'area'] = 5800
            df.loc[1358, 'area'] = 2660
            df.loc[3195, 'area'] = 2850
            df.loc[2131, 'area'] = 1812
            df.loc[3088, 'area'] = 2160
            df.loc[3444, 'area'] = 1175


            df.loc[2131, 'carpet_area'] = 1812
            df['price_per_sqft'] = round((df['price'] * 10000000) / df['area'])

            x = df[df['price_per_sqft'] <= 20000]
            min_room_area = (x['area'] / x['bedRoom']).quantile(0.02)
            df = df[(df['area'] / df['bedRoom']) >= min_room_area]
            
            return df
        
        except Exception as e:
              logging.error("Error occurred in Outlier Processing", exc_info=True)
              raise MyException(e, sys)

    def detect_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
         """
         Detects outliers in a given column of a DataFrame using the IQR method.
         
         Parameters:
             df (pd.DataFrame): The input DataFrame.
             column (str): The column name to analyze for outliers.
         
         Returns:
             pd.DataFrame: A DataFrame containing the detected outliers.
         """
         try:
             
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            
            return outliers
         except Exception as e:
              logging.error("Error occurred in Finding IQR", exc_info=True)
              raise MyException(e, sys)

class RemovingOutlier:
    """
    Feature engineering class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: OutlierStrategy) -> None:
        self.outlier_removed_data = data
        self.strategy = strategy

    def handle_outlier(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_outlier(self.outlier_removed_data)