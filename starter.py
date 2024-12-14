# Standard Library Imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import os
import random

# Third-Party Imports
import ee
import geemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from lightgbm import LGBMClassifier
from shapely.affinity import scale, translate
from skimage import exposure
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from shapely import wkt
import geopandas as gpd

import warnings
warnings.filterwarnings('ignore')

import logging
# Set up a logger to capture Rasterio warnings
logging.getLogger("rasterio._env").setLevel(logging.ERROR)

from my_configs.configs import Config
from my_utils.utils import Utils

rooth_path = Config.ROOT_PATH
train_path = Config.TRAIN_CSV
test_path = Config.TEST_CSV

# Authenticate with Google Earth Engine
# This opens a browser prompt for authentication, if not previously authenticated
ee.Authenticate()

# Initialize Earth Engine with a specific project
# Replace "project" with your project ID as needed
ee.Initialize(project=Config.PROJECT)


# Load the datasets
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Convert pandas DataFrames to GeoDataFrames with CRS set to 'epsg:4326'
train = gpd.GeoDataFrame(train, crs='epsg:4326', geometry=train['geometry'].apply(wkt.loads))
test = gpd.GeoDataFrame(test, crs='epsg:4326', geometry=test['geometry'].apply(wkt.loads))

# Concatenate train and test datasets into a single DataFrame for consistent processing
# 'dataset' column distinguishes between train and test rows
data = pd.concat(
    [train.assign(dataset='train'), test.assign(dataset='test')]
).reset_index(drop=True)

donwload = True

if donwload:
    error_indices = []  # Initialize list to collect indices of error rows
    # Execute image downloads for each row in parallel to improve performance
    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
        futures = [executor.submit(Utils.process_row, index, row) for index, row in data.iterrows()]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading Images", unit="image"): 
            index, image_file = future.result()
            if image_file:
                data.at[index, 'tif_path'] = image_file  # Store the downloaded file path
            else:
                error_indices.append(index)  # Collect indices of rows with errors
    
    # Save error rows to a separate CSV file
    if error_indices:
        error_rows = data.loc[error_indices]
        error_rows.to_csv('error_rows.csv', index=False)
        logging.info(f"Saved {len(error_indices)} error rows to error_rows.csv")
    else:
        logging.info("All images downloaded successfully without errors.")
    
    # Retry Mechanism for Error Rows
    if os.path.exists('error_rows.csv') and not error_indices:
        logging.info("No error rows to retry.")
    elif os.path.exists('error_rows.csv'):
        retry_data = pd.read_csv('error_rows.csv')
        retry_error_indices = []  # Initialize list for new error indices
        
        logging.info("Retrying failed image downloads from error_rows.csv...")
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            retry_futures = [executor.submit(Utils.process_row, index, row) for index, row in retry_data.iterrows()]
            for future in tqdm(as_completed(retry_futures), total=len(retry_futures), desc="Retrying Error Rows", unit="image"):
                index, image_file = future.result()
                if image_file:
                    data.at[index, 'tif_path'] = image_file  # Update the downloaded file path
                else:
                    retry_error_indices.append(index)  # Collect new errors
        
        # Save any new error rows back to CSV
        if retry_error_indices:
            new_error_rows = retry_data.loc[retry_error_indices]
            new_error_rows.to_csv('error_rows.csv', index=False)
            logging.info(f"Saved {len(retry_error_indices)} retry error rows to error_rows.csv")
        else:
            os.remove('error_rows.csv')  # Remove the file if all retries succeeded
            logging.info("All error rows re-downloaded successfully.")
else:
    # Load existing data with image paths if downloads are not required
    os.system('cp -r ./downloads .')
    data_path = "./data.csv"

    data = pd.read_csv(data_path)