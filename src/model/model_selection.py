from statistics import LinearRegression
import sys
import pickle
import numpy as np
import pandas as pd
import category_encoders as ce

from typing import Union
from abc import ABC, abstractmethod

from sklearn.metrics import mean_absolute_error

from src.logger import logging
from src.exception import MyException

from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


class ModelTrainingStrategy(ABC):

    @abstractmethod
    def handle_training(self, ddata: pd.DataFrame) -> pd.DataFrame:
        pass

class ModelTrainingConfig(ModelTrainingStrategy):

    def handle_training(self, data: pd.DataFrame) -> pd.DataFrame:

        try:
            df = data
            df['luxury_category'].fillna(df['luxury_category'].mode()[0], inplace=True)

            # Replace numeric furnishing_type values with categorical labels
            df['furnishing_type'] = df['furnishing_type'].replace({0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})

            # Define features and target variable
            X = df.drop(columns=['price'])
            print(X.info())
            y = np.log1p(df['price'])

            # Define numerical and categorical features
            num_features = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
            cat_features = ['property_type', 'sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

            # Fix: Handle unknown categories in encoding
            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

            # Create a column transformer for preprocessing
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', ordinal_encoder, cat_features),
                    ('cat1',OneHotEncoder(handle_unknown="ignore"), ['sector','agePossession']),
                    ('target_enc', ce.TargetEncoder(), ['sector'])  # Target encoding for 'sector'
                ],
                remainder='passthrough'
            )

            # Create pipeline
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(random_state=42))  # Added random_state for reproducibility
            ])

            # Define parameter grid
            param_grid = {
                'regressor__n_estimators': [50, 100, 200, 300],
                'regressor__max_depth': [None, 10, 20, 30],
                'regressor__max_samples': [0.1, 0.25, 0.5, 1.0],
                'regressor__max_features': ['sqrt', 'log2']  # Replaced 'auto' with 'sqrt' and 'log2'
            }

            # K-fold cross-validation
            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='r2', n_jobs=-1, verbose=4)
            search.fit(X, y)

            with open('pipeline.pkl', 'wb') as file:
                pickle.dump(pipeline, file)
            
            with open('df.pkl', 'wb') as file:
                pickle.dump(X, file)
        
        except Exception as e:
            logging.error("Error occurred while extracting zip file", exc_info=True)
            raise MyException(e, sys)

class ModelTraining(ModelTrainingStrategy):
    def __init__(self, data: pd.DataFrame, strategy: ModelTrainingStrategy):
        self.strategy = strategy
        self.df = data

    def handle_training(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_training(self.df)
