from pathlib import Path

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

"""
---------------------------------------------------------------
Training Pipeline related constant start with DATA_INGESTION VAR NAME
---------------------------------------------------------------
"""
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"
TRAINING_STAGE_NAME = "Training Pipeline"

"""
---------------------------------------------------------------
Environment Test and Dependency related constant 
---------------------------------------------------------------
"""
REQUIRED_PYTHON = "python3"
REQUIREMENTS_FILE = "requirements.txt"

"""
---------------------------------------------------------------
Data ingestion related constant 
---------------------------------------------------------------
"""
INGESTION_STAGE_NAME = "Data Ingestion"

"""
---------------------------------------------------------------
Data PrePrecessing related constant 
---------------------------------------------------------------
"""
PRE_PROCESSING_STAGE_NAME = "Data Pre-Processing"
N_CLUSTERS = 3
RANDOM_STATE = 42
WCSS_REDUCED = []

WEIGHTS = {'24/7 Power Backup': 8, '24/7 Water Supply': 4, '24x7 Security': 7, 'ATM': 4,
                       'Aerobics Centre': 6,
                       'Airy Rooms': 8, 'Amphitheatre': 7, 'Badminton Court': 7, 'Banquet Hall': 8,
                       'Bar/Chill-Out Lounge': 9,
                       'Barbecue': 7, 'Basketball Court': 7, 'Billiards': 7, 'Bowling Alley': 8, 'Business Lounge': 9,
                       'CCTV Camera Security': 8, 'Cafeteria': 6, 'Car Parking': 6, 'Card Room': 6,
                       'Centrally Air Conditioned': 9, 'Changing Area': 6, "Children's Play Area": 7, 'Cigar Lounge': 9,
                       'Clinic': 5, 'Club House': 9, 'Concierge Service': 9, 'Conference room': 8, 'Creche/Day care': 7,
                       'Cricket Pitch': 7, 'Doctor on Call': 6, 'Earthquake Resistant': 5, 'Entrance Lobby': 7,
                       'False Ceiling Lighting': 6, 'Feng Shui / Vaastu Compliant': 5, 'Fire Fighting Systems': 8,
                       'Fitness Centre / GYM': 8, 'Flower Garden': 7, 'Food Court': 6, 'Foosball': 5, 'Football': 7,
                       'Fountain': 7, 'Gated Community': 7, 'Golf Course': 10, 'Grocery Shop': 6, 'Gymnasium': 8,
                       'High Ceiling Height': 8, 'High Speed Elevators': 8, 'Infinity Pool': 9, 'Intercom Facility': 7,
                       'Internal Street Lights': 6, 'Internet/wi-fi connectivity': 7, 'Jacuzzi': 9, 'Jogging Track': 7,
                       'Landscape Garden': 8, 'Laundry': 6, 'Lawn Tennis Court': 8, 'Library': 8, 'Lounge': 8,
                       'Low Density Society': 7, 'Maintenance Staff': 6, 'Manicured Garden': 7, 'Medical Centre': 5,
                       'Milk Booth': 4, 'Mini Theatre': 9, 'Multipurpose Court': 7, 'Multipurpose Hall': 7,
                       'Natural Light': 8, 'Natural Pond': 7, 'Park': 8, 'Party Lawn': 8,
                       'Piped Gas': 7, 'Pool Table': 7, 'Power Back up Lift': 8, 'Private Garden / Terrace': 9,
                       'Property Staff': 7, 'RO System': 7, 'Rain Water Harvesting': 7, 'Reading Lounge': 8,
                       'Restaurant': 8,
                       'Salon': 8, 'Sauna': 9, 'Security / Fire Alarm': 9, 'Security Personnel': 9,
                       'Separate entry for servant room': 8, 'Sewage Treatment Plant': 6, 'Shopping Centre': 7,
                       'Skating Rink': 7, 'Solar Lighting': 6, 'Solar Water Heating': 7, 'Spa': 9,
                       'Spacious Interiors': 9,
                       'Squash Court': 8, 'Steam Room': 9, 'Sun Deck': 8,
                       'Swimming Pool': 8, 'Temple': 5, 'Theatre': 9, 'Toddler Pool': 7, 'Valet Parking': 9,
                       'Video Door Security': 9, 'Visitor Parking': 7, 'Water Softener Plant': 7, 'Water Storage': 7,
                       'Water purifier': 7, 'Yoga/Meditation Area': 7
                       }

"""
---------------------------------------------------------------
Model Selection related constant 
---------------------------------------------------------------
"""
MODEL_DICT = {
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

PARAM_GRID = {
    'regressor__n_estimators': [50, 100, 200, 300],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__max_samples':[0.1, 0.25, 0.5, 1.0],
    'regressor__max_features': ['auto', 'sqrt']
}

COLUMNS_TO_ENCODE = ['property_type','sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']