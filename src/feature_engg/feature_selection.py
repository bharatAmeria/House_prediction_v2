import os
import shap
import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from src.constants import *
from src.entity.config_entity import DataCleaningConfig
from src.logger import logging
from src.exception import MyException


class FeatureSelectionStrategy(ABC):
    """
    Abstract Class defining strategy for handling data
    """
    @abstractmethod
    def handle_FS(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class FeatureSelectionConfig(FeatureSelectionStrategy):


    def handle_FS(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering strategy which preprocesses the data.
        """
        try:
            df = data

            logging.info("Dropping unnecessary columns")
            df = df.drop(columns=COLUMNS_TO_DROP)

            logging.info("Categorizing luxury and floor levels")
            df['luxury_category'] = df['luxury_score'].apply(self.categorize_luxury)
            df['floor_category'] = df['floorNum'].apply(self.categorize_floor)
            df.drop(columns=['floorNum', 'luxury_score'], inplace=True)
            
            
            logging.info("Handling missing values using median imputation")
            num_imputer = SimpleImputer(strategy="median")
            num_cols = ["built_up_area"]
            df[num_cols] = num_imputer.fit_transform(df[num_cols])

            # Ensure the directory exists
            output_dir = "artifacts/model"
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv("artifacts/model/gurgaon_properties_post_feature_selection_v2.csv", index=False)

            data_label_encoded = df
            categorical_cols = df.select_dtypes(include=['object']).columns
            logging.info("Applying label encoding to categorical columns")
            for col in categorical_cols:
                oe = OrdinalEncoder()
                data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])
                logging.info(f"Categories for {col}: {oe.categories_}")

            logging.info("Splitting dataset into features and target variable")
            X_label = data_label_encoded.drop('price', axis=1)
            y_label = data_label_encoded['price']
            y_label = num_imputer.fit_transform(y_label.values.reshape(-1, 1)).ravel()

            logging.info("Computing correlation analysis")
            fi_df1 = data_label_encoded.corr()['price'].iloc[1:].to_frame().reset_index().rename(columns={'index': 'feature', 'price': 'corr_coeff'})

            logging.info("Training Random Forest Regressor for feature importance")
            rf_label = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
            rf_label.fit(X_label, y_label)
            fi_df2 = pd.DataFrame({'feature': X_label.columns, 'rf_importance': rf_label.feature_importances_}).sort_values(by='rf_importance', ascending=False)

            logging.info("Training Gradient Boosting Regressor for feature importance")
            gb_label = GradientBoostingRegressor()
            gb_label.fit(X_label, y_label)
            fi_df3 = pd.DataFrame({'feature': X_label.columns, 'gb_importance': gb_label.feature_importances_}).sort_values(by='gb_importance', ascending=False)

            logging.info("Computing Permutation Importance")
            X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X_label, y_label, test_size=TEST_SIZE, random_state=RANDOM_STATE)
            rf_label.fit(X_train_label, y_train_label)
            perm_importance = permutation_importance(rf_label, X_test_label, y_test_label, n_repeats=30, random_state=RANDOM_STATE)
            fi_df4 = pd.DataFrame({'feature': X_label.columns, 'permutation_importance': perm_importance.importances_mean}).sort_values(by='permutation_importance', ascending=False)

            logging.info("Applying LASSO Regression for feature selection")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_label)
            lasso = Lasso(alpha=ALPHA, random_state=RANDOM_STATE)
            lasso.fit(X_scaled, y_label)
            fi_df5 = pd.DataFrame({'feature': X_label.columns, 'lasso_coeff': lasso.coef_}).sort_values(by='lasso_coeff', ascending=False)

            logging.info("Applying Recursive Feature Elimination (RFE)")
            selector_label = RFE(ESTIMATOR, n_features_to_select=X_label.shape[1], step=1)
            selector_label = selector_label.fit(X_label, y_label)
            selected_features = X_label.columns[selector_label.support_]
            selected_coefficients = selector_label.estimator_.feature_importances_
            fi_df6 = pd.DataFrame({'feature': selected_features, 'rfe_score': selected_coefficients}).sort_values(by='rfe_score', ascending=False)

            logging.info("Training Linear Regression for feature importance")
            lin_reg = LinearRegression()
            lin_reg.fit(X_scaled, y_label)
            fi_df7 = pd.DataFrame({'feature': X_label.columns, 'reg_coeffs': lin_reg.coef_}).sort_values(by='reg_coeffs', ascending=False)

            logging.info("Computing SHAP values for feature importance")
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_label, y_label)
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_label)
            shap_sum = np.abs(shap_values).mean(axis=0)
            fi_df8 = pd.DataFrame({'feature': X_label.columns, 'SHAP_score': shap_sum}).sort_values(by='SHAP_score', ascending=False)

            logging.info("Merging all feature importance dataframes")
            final_fi_df = fi_df1.merge(fi_df2, on='feature').merge(fi_df3, on='feature').merge(fi_df4, on='feature').merge(fi_df5, on='feature').merge(fi_df6, on='feature').merge(fi_df7, on='feature').merge(fi_df8, on='feature').set_index('feature')
            final_fi_df = final_fi_df.divide(final_fi_df.sum(axis=0), axis=1)
            final_fi_df[['rf_importance', 'gb_importance', 'permutation_importance', 'rfe_score', 'SHAP_score']].mean(axis=1).sort_values(ascending=False)

            logging.info("Performing cross-validation with Random Forest")
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            scores1 = cross_val_score(rf, X_label, y_label, cv=5, scoring='r2')
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            scores2 = cross_val_score(rf, X_label.drop(columns=['pooja room', 'study room', 'others']), y_label, cv=5, scoring='r2')

            logging.info(f"Cross-validation scores (full dataset): {scores1}")
            logging.info(f"Cross-validation scores (reduced dataset): {scores2}")
            logging.info(f"Final Feature Importance Table: {final_fi_df}")

            export_df = X_label.drop(columns=['pooja room', 'study room', 'others'])
            export_df['price'] = y_label

            # Ensure the directory exists
            output_dir = "artifacts/model"
            os.makedirs(output_dir, exist_ok=True)
            df.to_csv("artifacts/model/gurgaon_properties_post_feature_selection.csv", index=False)
        except MyException as e:
            raise e
        
    def categorize_luxury(self, score):
        if 0 <= score < 50:
            return "Low"
        elif 50 <= score < 150:
            return "Medium"
        elif 150 <= score <= 175:
            return "High"
        else:
            return None 
        
    def categorize_floor(self, floor):
        if 0 <= floor <= 2:
            return "Low Floor"
        elif 3 <= floor <= 10:
            return "Mid Floor"
        elif 11 <= floor <= 51:
            return "High Floor"
        else:
            return None  
        
class FeatureSelection(FeatureSelectionStrategy):
    """
    Feature engineering class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: FeatureSelectionStrategy, config: DataCleaningConfig) -> None:
        self.config = config
        self.df = data
        self.strategy = strategy

    def handle_FS(self) -> Union[pd.DataFrame, pd.Series]:
        return self.strategy.handle_FS(self.df)
        
