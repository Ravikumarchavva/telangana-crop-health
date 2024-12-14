
import os
from pathlib import Path

# Project root path
ROOT_PATH = Path(".")

# Paths to data files
TRAIN_CSV = ROOT_PATH / 'train.csv'
TEST_CSV = ROOT_PATH / 'test.csv'
SUBMISSION_CSV = ROOT_PATH / "SampleSubmission.csv"

# Output directories
OUTPUT_FOLDER = 'downloads'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Environment variables
PROJECT = os.getenv("PROJECT")
DOWNLOAD_DATA = os.getenv("DOWNLOAD")

# Scaling factor for geometries
SCALE_FACTOR = 5

# Earth Engine and image settings
IMAGE_COLLECTION_NAME = 'COPERNICUS/S2'
BANDS = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']
SENTINEL_SCALE = 10  # Sentinel-2 resolution in meters

# Parallel processing settings
MAX_WORKERS = 100

# LightGBM configuration
LIGHTGBM_CONFIG = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': None,  # To be set dynamically based on data
    'num_leaves': 81,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'random_state': 42,
    'force_row_wise': 'true'
}