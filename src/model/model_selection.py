import os
import sys
import pickle
import numpy as np
import pandas as pd
import category_encoders as ce

from typing import Union
from abc import ABC, abstractmethod

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
            df['luxury_category'].fillna(df['luxury_category'].mode()[0], inplace=True)

            # Replace numeric furnishing_type values with categorical labels
            df['furnishing_type'] = df['furnishing_type'].replace({0.0: 'unfurnished', 1.0: 'semifurnished', 2.0: 'furnished'})

            # Define features and target variable
            self.X = df.drop(columns=['price'])
            print(self.X.info())
            self.y = np.log1p(df['price'])

            # Define numerical and categorical features
            num_features = ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']
            cat_features = ['property_type', 'sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']


            model_dict = {
                        'linear_reg':LinearRegression(),
                        'svr':SVR(),
                        'ridge':Ridge(),
                        'LASSO':Lasso(),
                        'decision tree': DecisionTreeRegressor(),
                        'random forest':RandomForestRegressor(),
                        'extra trees': ExtraTreesRegressor(),
                        'gradient boosting': GradientBoostingRegressor(),
                        'adaboost': AdaBoostRegressor(),
                        'mlp': MLPRegressor(),
                        'xgboost':XGBRegressor()
}
            # Creating a column transformer for preprocessing
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
                    ('cat', OrdinalEncoder(), cat_features)
                ], 
                remainder='passthrough'
            )

            # Creating a pipeline
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', LinearRegression())
            ])

            # K-fold cross-validation
            kfold = KFold(n_splits=10, shuffle=True, random_state=42)
            scores = cross_val_score(pipeline, self.X, self.y, cv=kfold, scoring='r2')

            print(scores.mean(),scores.std())

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,test_size=0.2,random_state=42)

            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_test)
            mean_absolute_error(np.expm1(y_test),y_pred)


            model_output = []
            for model_name,model in model_dict.items():
                model_output.append(self.scorer(model_name, model))

            print(model_output)

            # Ensure the directory exists
            output_dir = "artifacts/model"
            os.makedirs(output_dir, exist_ok=True)
            with open('artifacts/model/pipeline.pkl', 'wb') as file:
                pickle.dump(pipeline, file)
            
            # Ensure the directory exists
            output_dir = "artifacts/model"
            os.makedirs(output_dir, exist_ok=True)
            with open('artifacts/model/df.pkl', 'wb') as file:
                pickle.dump(self.X, file)
        
        except Exception as e:
            logging.error("Error occurred while extracting zip file", exc_info=True)
            raise MyException(e, sys)
        
    def scorer(self, model_name, model):
    
        output = []
        
        output.append(model_name)
        
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', model)
        ])
        
        # K-fold cross-validation
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        scores = cross_val_score(pipeline, self.X, self.y, cv=kfold, scoring='r2')
        
        output.append(scores.mean())
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y,test_size=0.2,random_state=42)
        
        pipeline.fit(X_train,y_train)
        
        y_pred = pipeline.predict(X_test)
        
        y_pred = np.expm1(y_pred)
        
        output.append(mean_absolute_error(np.expm1(y_test),y_pred))
        
        return output

class ModelTraining(ModelTrainingStrategy):
    def __init__(self, data: pd.DataFrame, strategy: ModelTrainingStrategy):
        self.strategy = strategy
        self.df = data

    def handle_training(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        return self.strategy.handle_training(self.df)
