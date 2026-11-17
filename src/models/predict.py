import joblib
import pandas as pd

from pathlib import Path
from functools import lru_cache

from src.features.feature_engineering import engineer_features
from src.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT /
    "artifacts" /
    "best_model.joblib"
)


# --------------------------------------------------
# REQUIRED CUSTOMER FEATURES
# --------------------------------------------------

REQUIRED_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges"
]


# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run the model selection script first."
        )

    logger.info("Loading selected churn prediction model.")

    model = joblib.load(MODEL_PATH)

    logger.info("Trained model loaded successfully.")

    return model


# --------------------------------------------------
# VALIDATE CUSTOMER INPUT
# --------------------------------------------------

def validate_customer_data(customer_data):

    if not isinstance(customer_data, dict):

        raise TypeError(
            "Customer data must be provided as a dictionary."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in customer_data
    ]

    if missing_columns:

        raise ValueError(
            f"Missing customer fields: {missing_columns}"
        )

    customer_df = pd.DataFrame(
        [customer_data],
        columns=REQUIRED_COLUMNS
    )

    return customer_df


# --------------------------------------------------
# CUSTOMER CHURN PREDICTION
# --------------------------------------------------

def predict_customer_churn(customer_data, model=None):

    logger.info("Starting customer churn prediction.")

    # Validate customer input
    customer_df = validate_customer_data(
        customer_data
    )

    # Convert TotalCharges to numeric
    customer_df["TotalCharges"] = pd.to_numeric(
        customer_df["TotalCharges"],
        errors="coerce"
    )

    # Apply feature engineering
    customer_df = engineer_features(
        customer_df
    )

    # Load trained model when one is not supplied
    if model is None:

        model = load_model()

    # Generate churn probability
    probability = float(
        model.predict_proba(customer_df)[0][1]
    )

    # Initial classification threshold
    threshold = 0.50

    # Generate binary prediction using the threshold
    prediction = int(
        probability >= threshold
    )

    churn_prediction = (
        "Yes" if prediction == 1 else "No"
    )

    # Retention risk classification
    if probability < 0.30:

        risk_category = "Low"

    elif probability < 0.60:

        risk_category = "Medium"

    else:

        risk_category = "High"

    result = {

        "churn_prediction": churn_prediction,

        "churn_probability": round(
            probability,
            4
        ),

        "prediction": prediction,

        "risk_category": risk_category,

        "classification_threshold": threshold

    }

    logger.info(
        "Customer churn prediction completed."
    )

    return result


# --------------------------------------------------
# TEST EXECUTION
# --------------------------------------------------

def main():

    sample_customer = {

        "gender": "Female",

        "SeniorCitizen": 0,

        "Partner": "Yes",

        "Dependents": "No",

        "tenure": 12,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "Yes",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "Yes",

        "StreamingMovies": "Yes",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 85.50,

        "TotalCharges": 1026.00

    }

    result = predict_customer_churn(
        sample_customer
    )

    print("\n" + "=" * 55)

    print("CUSTOMER CHURN PREDICTION RESULT")

    print("=" * 55)

    for key, value in result.items():

        print(f"{key}: {value}")


if __name__ == "__main__":

    main()