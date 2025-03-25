import re
import sys
import pandas as pd
import ast
import pickle
from wordcloud import WordCloud
from abc import ABC, abstractmethod
from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataCleaningConfig

class DataVisualizationStrategy(ABC):
    @abstractmethod
    def handle_viz(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class FlatsDataPreProcessingStrategy(DataVisualizationStrategy):

    def handle_viz(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data

            df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')

            latlong = pd.read_csv('latlong.csv')
            latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')
            latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')
            new_df = df.merge(latlong, on='sector')
            group_df = new_df.groupby('sector').mean()[['price','price_per_sqft','built_up_area','latitude','longitude']]

            new_df.to_csv('data_viz1.csv',index=False)

# word cloud
            df1 = pd.read_csv('gurgaon_properties.csv')
            wordcloud_df = df1.merge(df, left_index=True, right_index=True)[['features','sector']]

            main = []
            for item in wordcloud_df['features'].dropna().apply(ast.literal_eval):
                main.extend(item)

            feature_text = ' '.join(main)
            pickle.dump(feature_text, open('feature_text.pkl','wb'))