import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

class Config:
    # Project root path
    ROOT_PATH = Path(os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))

    # Paths to data files
    TRAIN_CSV = ROOT_PATH / 'telangana_data' / 'data' / 'train.csv'
    TEST_CSV = ROOT_PATH / 'telangana_data' / 'data' / 'test.csv'
    SUBMISSION_CSV = ROOT_PATH / 'telangana_data' / 'data' / "SampleSubmission.csv"

    # Download directory
    DOWNLOAD_FOLDER = 'downloads'
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    # Environment variables
    PROJECT = os.getenv("PROJECT_ID", "zinditelangana2024")  # Default project ID
    DOWNLOAD_DATA = os.getenv("DOWNLOAD", "True")

    # Scaling factor for geometries
    SCALE_FACTOR = 5

    # Earth Engine and image settings
    IMAGE_COLLECTION_NAME = 'COPERNICUS/S2'
    BANDS = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
    SENTINEL_SCALE = 10  # Sentinel-2 resolution in meters

    # Parallel processing settings
    MAX_WORKERS = 50

    # LightGBM configuration
    LIGHTGBM_CONFIG = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': 'multi_logloss',
        'num_class': 3,  # Set a default or handle dynamically based on dataset
        'num_leaves': 81,
        'learning_rate': 0.01,
        'n_estimators': 500,
        'random_state': 42,
        'force_row_wise': True  # Use boolean instead of string
    }