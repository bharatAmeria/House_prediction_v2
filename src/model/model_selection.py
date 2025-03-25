import sys
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.logger import logging
from src.exception import MyException
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA

class ModelSelectionStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class ModelSelectionConfig(ModelSelectionStrategy):

    def handle_selection(self, data: pd.DataFrame) -> pd.DataFrame:

        try:
            df = data

            df['furnishing_type'] = df['furnishing_type'].replace({0.0:'unfurnished',1.0:'semifurnished',2.0:'furnished'})
            
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

            model_output = []
            for model_name,model in model_dict.items():
                model_output.append(self.scorer(model_name, model))

            print(model_output)
            model_df = pd.DataFrame(model_output, columns=['name','r2','mae'])
            print(model_df.sort_values(['mae']))
                    
        except Exception as e:
            logging.error("Error", exc_info=True)
            raise MyException(e, sys)
     
    def scorer(self, data: pd.DataFrame, model_name, model):
       
       df = data
       # Ordinal Encoding
       columns_to_encode = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']
            
       X = df.drop(columns=['price'])
       y = df['price']  

       # Applying the log1p transformation to the target variable
       y_transformed = np.log1p(y)

       # Creating a column transformer for preprocessing
       preprocessor = ColumnTransformer(
           transformers=[
               ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
               ('cat', OrdinalEncoder(), columns_to_encode)
           ], 
           remainder='passthrough'
       )

       output = []
       
       output.append(model_name)
       
       pipeline = Pipeline([
           ('preprocessor', preprocessor),
           ('regressor', model)
       ])
       
       # K-fold cross-validation
       kfold = KFold(n_splits=10, shuffle=True, random_state=42)
       scores = cross_val_score(pipeline, X, y_transformed, cv=kfold, scoring='r2')
       
       output.append(scores.mean())
       
       X_train, X_test, y_train, y_test = train_test_split(X,y_transformed,test_size=0.2,random_state=42)
       
       pipeline.fit(X_train,y_train)
       
       y_pred = pipeline.predict(X_test)
       
       y_pred = np.expm1(y_pred)
       
       output.append(mean_absolute_error(np.expm1(y_test),y_pred))
       
       return output
           
