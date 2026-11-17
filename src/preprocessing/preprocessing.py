import pandas as pd
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the customer churn dataset.

    Operations:
    - Create a copy of the original dataset.
    - Remove unnecessary whitespace.
    - Convert TotalCharges to numeric.
    - Handle invalid values.
    - Validate target variable.
    - Remove duplicate customer records.
    """

    logger.info("Starting customer data cleaning.")

    data = df.copy()

    initial_rows = len(data)

    # 1. Remove leading and trailing whitespace
    object_columns = data.select_dtypes(
        include=["object"]
    ).columns

    for column in object_columns:
        data[column] = data[column].str.strip()

    # 2. Convert TotalCharges into numeric
    if "TotalCharges" in data.columns:

        data["TotalCharges"] = pd.to_numeric(
            data["TotalCharges"],
            errors="coerce"
        )

    # 3. Replace infinite values
    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # 4. Validate target variable
    if "Churn" not in data.columns:
        raise ValueError("Target column 'Churn' is missing.")

    data = data.dropna(subset=["Churn"])

    invalid_targets = ~data["Churn"].isin(["Yes", "No"])

    if invalid_targets.any():
        raise ValueError("Unexpected values found in Churn column.")

    # 5. Remove duplicate customer records
    if "customerID" in data.columns:

        duplicate_count = data["customerID"].duplicated().sum()

        if duplicate_count > 0:
            logger.warning(
                "%s duplicate customer IDs found.",
                duplicate_count
            )

            data = data.drop_duplicates(
                subset=["customerID"],
                keep="first"
            )

    # 6. Reset index
    data = data.reset_index(drop=True)

    removed_rows = initial_rows - len(data)

    logger.info(
        "Data cleaning completed. Removed rows: %s",
        removed_rows
    )

    logger.info(
        "Final dataset shape: %s",
        data.shape
    )

    return data

# ==================================================
# MACHINE LEARNING PREPROCESSING PIPELINE
# ==================================================

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(X):
    """
    Build a reusable preprocessing pipeline.

    Numerical features:
        - Median imputation
        - Standard scaling

    Categorical features:
        - Most frequent imputation
        - One-hot encoding

    Parameters:
        X: Training feature DataFrame

    Returns:
        Configured ColumnTransformer
    """

    logger.info("Building preprocessing pipeline.")

    # Identify numerical columns
    numerical_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Identify categorical columns
    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    logger.info(
        "Numerical features: %s",
        numerical_features
    )

    logger.info(
        "Categorical features: %s",
        categorical_features
    )

    # Numerical preprocessing
    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # Categorical preprocessing
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    # Combine preprocessing pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    logger.info("Preprocessing pipeline created successfully.")

    return preprocessor