import polars as pl

train_data = pl.read_csv("./train.csv", has_header=True)
test_data = pl.read_csv("./test.csv", has_header=True)

train_data = train_data.drop("State")
test_data = test_data.drop("State")
train_data = train_data.with_columns(pl.col("category").alias("Target")).drop("category")
test_data = test_data.with_columns(pl.col("category").alias("Target")).drop("category")

train_data = train_data.with_columns(pl.col("SDate").str.to_date("%Y-%m-%d %H:%M:%S").alias("SDate"))
train_data = train_data.with_columns(pl.col("HDate").str.to_date("%Y-%m-%d %H:%M:%S").alias("HDate"))

test_data = test_data.with_columns(pl.col("SDate").str.to_date("%Y-%m-%d %H:%M:%S").alias("SDate"))
test_data = test_data.with_columns(pl.col("HDate").str.to_date("%Y-%m-%d %H:%M:%S").alias("HDate"))

import pandas as pd
train_new_features_df = pl.read_csv("train_new_features.csv").drop("index")
test_new_features_df = pl.read_csv("test_new_features.csv").drop("index")

train_data = train_data.hstack(train_new_features_df)
test_data = test_data.hstack(test_new_features_df)

train_data = train_data.drop("SDate", "HDate", "geometry", "tif_path").fill_null(strategy="min")
test_data = test_data.drop("SDate", "HDate", "geometry", "tif_path").fill_null(strategy="min")

X = train_data.drop("FarmID", 'Target').to_pandas()
y = train_data.select("Target").to_pandas()

from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

# Replace inf and -inf with NaN
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_valid = X_valid.replace([np.inf, -np.inf], np.nan)

# Define features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
categorical_features_indices = [X.columns.get_loc(col) for col in categorical_features]

# Impute missing values in numeric and categorical features
numeric_imputer = SimpleImputer(strategy='mean')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Impute numeric features
X_train_numeric = X_train[numeric_features]
X_train_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(X_train_numeric), columns=numeric_features)

# Impute categorical features
X_train_categorical = X_train[categorical_features]
X_train_categorical_imputed = pd.DataFrame(categorical_imputer.fit_transform(X_train_categorical), columns=categorical_features)

# Combine imputed features
X_train_imputed = pd.concat([X_train_numeric_imputed, X_train_categorical_imputed], axis=1)

from imblearn.over_sampling import SMOTENC

# Apply SMOTENC to the imputed training data
smote = SMOTENC(sampling_strategy='minority', categorical_features=categorical_features_indices, random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

# Preprocessing for numeric features
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessor (scale numeric features, pass through categorical features)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ],
    remainder='passthrough'
)

# Preprocess the resampled training data and validation data
X_train_preprocessed = preprocessor.fit_transform(X_train_resampled)
X_valid_imputed_numeric = pd.DataFrame(numeric_imputer.transform(X_valid[numeric_features]), columns=numeric_features)
X_valid_imputed_categorical = pd.DataFrame(categorical_imputer.transform(X_valid[categorical_features]), columns=categorical_features)
X_valid_imputed = pd.concat([X_valid_imputed_numeric, X_valid_imputed_categorical], axis=1)
X_valid_preprocessed = preprocessor.transform(X_valid_imputed)

# Fit the classifier
clf = CatBoostClassifier(iterations=1000, verbose=200, cat_features=categorical_features_indices)
clf.fit(X_train_preprocessed, y_train_resampled, eval_set=(X_valid_preprocessed, y_valid))