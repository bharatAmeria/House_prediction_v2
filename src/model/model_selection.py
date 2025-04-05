import os
import sys
import pickle
import numpy as np
import pandas as pd
import category_encoders as ce

from typing import Union
from abc import ABC, abstractmethod

from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import MyException
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor


class ModelTrainingStrategy(ABC):

    @abstractmethod
    def handle_training(self, ddata: pd.DataFrame) -> pd.DataFrame:
        pass

class ModelTrainingConfig(ModelTrainingStrategy):

    def handle_training(self, data: pd.DataFrame) -> pd.DataFrame:

        try:
            df = data
            print(df.head())

            param_grid = {
                        'regressor__n_estimators': [300],
                        'regressor__max_depth': [20],
                        'regressor__max_samples':[1.0],
                        'regressor__max_features': ['sqrt']
                    }
            df['luxury_category'].fillna(df['luxury_category'].mode()[0], inplace=True)

            # Replace numeric furnishing_type values with categorical labels
            df['furnishing_type'] = df['furnishing_type'].replace({0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})

            # Define features and target variable
            X = df.drop(columns=['price'])
            y = np.log1p(df['price'])

            # Define numerical and categorical features
            cat_features = ['property_type', 'sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']

            # Creating a column transformer for preprocessing
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
                    ('cat', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_features),
                    ('cat1',OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False),['agePossession']),
                    ('target_enc', ce.TargetEncoder(), ['sector'])
                ], 
                remainder='passthrough'
            )

            # Creating a pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=500)),
            ])

            # K-fold cross-validation
            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            search = GridSearchCV(pipeline, param_grid, cv=kfold, scoring='r2', n_jobs=-1, verbose=4)

            search.fit(X, y)
            final_pipe = search.best_estimator_

            print(search.best_params_,search.best_score_)
            final_pipe.fit(X, y)

            output_dir = "artifacts/model"
            os.makedirs(output_dir, exist_ok=True)

            with open(os.path.join(output_dir, 'pipeline.pkl'), 'wb') as file:
                pickle.dump(pipeline, file)

            with open(os.path.join(output_dir, 'df.pkl'), 'wb') as file:
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
