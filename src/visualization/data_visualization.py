import sys
import pandas as pd
import ast
import pickle
from wordcloud import WordCloud
from abc import ABC, abstractmethod
from src.logger import logging
from typing import Union
from src.exception import MyException
from src.entity.config_entity import DataVisualizationConfig
from src.config.configuration import ConfigurationManager

class DataVisualizationStrategy(ABC):
    @abstractmethod
    def handle_viz(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class Datavisualization(DataVisualizationStrategy):

    def handle_viz(self, data: pd.DataFrame, config=ConfigurationManager()) -> pd.DataFrame:
        try:
            config = config.get_data_visualization_config()
            df = data

            latlong = pd.read_csv(config.latlong)
            latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')
            latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')
            new_df = df.merge(latlong, on='sector')
            group_df = new_df.groupby('sector').mean()[['price','price_per_sqft','built_up_area','latitude','longitude']]

            new_df.to_csv(config.data_viz ,index=False)
            
            # word cloud
            df1 = pd.read_csv(config.gurgaon_properties)
            wordcloud_df = df1.merge(df, left_index=True, right_index=True)[['features','sector']]

            main = []
            for item in wordcloud_df['features'].dropna().apply(ast.literal_eval):
                main.extend(item)

            feature_text = ' '.join(main)
            pickle.dump(feature_text, open(config.feature_text,'wb'))

        except Exception as e:
            logging.error("Error occurred in Data Visulaization", exc_info=True)
            raise MyException(e, sys)
        
class DataVisualization(DataVisualizationStrategy):
    """
    Data cleaning class which preprocesses the data
    """
    def __init__(self, data: pd.DataFrame, strategy: DataVisualizationStrategy, config = DataVisualizationConfig) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.config = config
        self.df = data
        self.strategy = strategy

    def handle_viz(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_viz(self.df)