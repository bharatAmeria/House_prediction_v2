import ast
import pickle
import re
import sys
import json
from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler

from src.config.configuration import ConfigurationManager
from src.entity.config_entity import RecommendSysConfig
from src.logger import logging
from src.exception import MyException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecommendStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_recommend(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class RecommenderSystemConfig(RecommendStrategy):
    """
    Data preprocessing strategy which preprocesses the data.
    """
    def handle_recommend(self, data: pd.DataFrame, config=ConfigurationManager()) -> pd.DataFrame:
        """
        Removes columns which are not required, fills missing values with median average values,
        and converts the data type to float.
        """
        try:
            config = config.get_recommend_sys_config()

            df = data
            
            # Initialize the scaler
            scaler = StandardScaler()
            df = df.drop(22)
            df[['PropertyName','TopFacilities']]['TopFacilities'][0]

            df['TopFacilities'] = df['TopFacilities'].apply(self.extract_list)
            df['FacilitiesStr'] = df['TopFacilities'].apply(' '.join)

            tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['FacilitiesStr'])
            tfidf_matrix.toarray()[0]

            self.cosine_sim1 = cosine_similarity(tfidf_matrix, tfidf_matrix)
            pickle.dump(self.cosine_sim1, open(config.cosine1, 'wb'))

            df[['PropertyName','PriceDetails']]['PriceDetails'][1]

            # Apply the refined parsing and generate the new DataFrame structure
            data_refined = []

            for _, row in df.iterrows():
                features = self.refined_parse_modified_v2(row['PriceDetails'])
                
                # Construct a new row for the transformed dataframe
                new_row = {'PropertyName': row['PropertyName']}
                
                # Populate the new row with extracted features
                for config in ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK', '6 BHK', '1 RK', 'Land']:
                    new_row[f'building type_{config}'] = features.get(f'building type_{config}')
                    new_row[f'area low {config}'] = features.get(f'area low {config}')
                    new_row[f'area high {config}'] = features.get(f'area high {config}')
                    new_row[f'price low {config}'] = features.get(f'price low {config}')
                    new_row[f'price high {config}'] = features.get(f'price high {config}')
                
                data_refined.append(new_row)

            df_final_refined_v2 = pd.DataFrame(data_refined).set_index('PropertyName')
            df_final_refined_v2['building type_Land'] = df_final_refined_v2['building type_Land'].replace({'':'Land'})

            categorical_columns = df_final_refined_v2.select_dtypes(include=['object']).columns.tolist()

            ohe_df = pd.get_dummies(df_final_refined_v2, columns=categorical_columns, drop_first=True)
            ohe_df.fillna(0,inplace=True)


            # Apply the scaler to the entire dataframe
            self.ohe_df_normalized = pd.DataFrame(scaler.fit_transform(ohe_df), columns=ohe_df.columns, index=ohe_df.index)

            # Compute the cosine similarity matrix
            self.cosine_sim2 = cosine_similarity(self.ohe_df_normalized)
            pickle.dump(self.cosine_sim2, open("artifacts/model/cosine_sim2", 'wb'))

            # Extract distances for each location
            location_matrix = {}
            for index, row in df.iterrows():
                distances = {}
                for location, distance in ast.literal_eval(row['LocationAdvantages']).items():
                    distances[location] = self.distance_to_meters(distance)
                location_matrix[index] = distances

            # Convert the dictionary to a dataframe
            location_df = pd.DataFrame.from_dict(location_matrix, orient='index')
            location_df.fillna(54000,inplace=True)

            # Apply the scaler to the entire dataframe
            self.location_df_normalized = pd.DataFrame(scaler.fit_transform(location_df), columns=location_df.columns, index=location_df.index)
            self.cosine_sim3 = cosine_similarity(self.location_df_normalized)
            pickle.dump(self.cosine_sim3, open("artifacts/model/cosine_sim3", 'wb'))

        except MyException as e:
            logging.error("Error occurred in Data Visulaization", exc_info=True)
            raise MyException(e, sys)
        
    def distance_to_meters(self, distance_str):
        try:
            if 'Km' in distance_str or 'KM' in distance_str:
                return float(distance_str.split()[0]) * 1000
            elif 'Meter' in distance_str or 'meter' in distance_str:
                return float(distance_str.split()[0])
            else:
                return None
        except:
            return None
        
    def recommend_properties_with_scores(self, property_name, top_n=247):
        
        cosine_sim_matrix = 30 * self.cosine_sim1 + 20 * self.cosine_sim2 + 8 * self.cosine_sim3
        # cosine_sim_matrix = cosine_sim3
        
        # Get the similarity scores for the property using its name as the index
        sim_scores = list(enumerate(cosine_sim_matrix[self.location_df_normalized.index.get_loc(property_name)]))
        
        # Sort properties based on the similarity scores
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the indices and scores of the top_n most similar properties
        top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
        
        # Retrieve the names of the top properties using the indices
        top_properties = self.location_df_normalized.index[top_indices].tolist()
        
        # Create a dataframe with the results
        recommendations_df = pd.DataFrame({
            'PropertyName': top_properties,
            'SimilarityScore': top_scores
        })
        
        return recommendations_df
        
    def recommend_properties_with_scores(self, property_name, top_n=247):
        # Get the similarity scores for the property using its name as the index
        sim_scores = list(enumerate(self.cosine_sim2[self.ohe_df_normalized.index.get_loc(property_name)]))
        
        # Sort properties based on the similarity scores
        sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get the indices and scores of the top_n most similar properties
        top_indices = [i[0] for i in sorted_scores[1:top_n+1]]
        top_scores = [i[1] for i in sorted_scores[1:top_n+1]]
        
        # Retrieve the names of the top properties using the indices
        top_properties = self.ohe_df_normalized.index[top_indices].tolist()
        
        # Create a dataframe with the results
        recommendations_df = pd.DataFrame({
            'PropertyName': top_properties,
            'SimilarityScore': top_scores
        })
        
        return recommendations_df
            
    # Function to parse and extract the required features from the PriceDetails column
    def refined_parse_modified_v2(self, detail_str):
          try:
              details = json.loads(detail_str.replace("'", "\""))
          except:
              return {}

          extracted = {}
          for bhk, detail in details.items():
              # Extract building type
              extracted[f'building type_{bhk}'] = detail.get('building_type')

              # Parsing area details
              area = detail.get('area', '')
              area_parts = area.split('-')
              if len(area_parts) == 1:
                  try:
                      value = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                      extracted[f'area low {bhk}'] = value
                      extracted[f'area high {bhk}'] = value
                  except:
                      extracted[f'area low {bhk}'] = None
                      extracted[f'area high {bhk}'] = None
              elif len(area_parts) == 2:
                  try:
                      extracted[f'area low {bhk}'] = float(area_parts[0].replace(',', '').replace(' sq.ft.', '').strip())
                      extracted[f'area high {bhk}'] = float(area_parts[1].replace(',', '').replace(' sq.ft.', '').strip())
                  except:
                      extracted[f'area low {bhk}'] = None
                      extracted[f'area high {bhk}'] = None

              # Parsing price details
              price_range = detail.get('price-range', '')
              price_parts = price_range.split('-')
              if len(price_parts) == 2:
                  try:
                      extracted[f'price low {bhk}'] = float(price_parts[0].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                      extracted[f'price high {bhk}'] = float(price_parts[1].replace('₹', '').replace(' Cr', '').replace(' L', '').strip())
                      if 'L' in price_parts[0]:
                          extracted[f'price low {bhk}'] /= 100
                      if 'L' in price_parts[1]:
                          extracted[f'price high {bhk}'] /= 100
                  except:
                      extracted[f'price low {bhk}'] = None
                      extracted[f'price high {bhk}'] = None

          return extracted
        
    def recommend_properties(self, df, property_name):
       # Get the index of the property that matches the name
       idx = df.index[df['PropertyName'] == property_name].tolist()[0]

       # Get the pairwise similarity scores with that property
       sim_scores = list(enumerate(self.cosine_sim1[idx]))

       # Sort the properties based on the similarity scores
       sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

       # Get the scores of the 10 most similar properties
       sim_scores = sim_scores[1:6]

       # Get the property indices
       property_indices = [i[0] for i in sim_scores]
       
       recommendations_df = pd.DataFrame({
           'PropertyName': df['PropertyName'].iloc[property_indices],
           'SimilarityScore': sim_scores
       })

       # Return the top 10 most similar properties
       return recommendations_df
           
    def extract_list(self, s):
        return re.findall(r"'(.*?)'", s)
        
class RecommenderSystem(RecommendStrategy):
    """
    Data cleaning class which preprocesses the data
    """
    def __init__(self, data: pd.DataFrame, strategy: RecommendStrategy, config=RecommendSysConfig) -> None:
        """Initializes the DataCleaning class with a specific strategy."""
        self.config = config
        self.df = data
        self.strategy = strategy

    def handle_recommend(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_recommend(self.df)
    

            