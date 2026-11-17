import json
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

from src.data.load_data import load_dataset

from src.preprocessing.preprocessing import (
    clean_customer_data,
    build_preprocessor
)

from src.features.feature_engineering import engineer_features

from src.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_PATH = PROJECT_ROOT / "artifacts"

ARTIFACT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_training_data():

    logger.info("Loading customer churn dataset.")

    df = load_dataset()

    logger.info("Cleaning customer data.")

    df = clean_customer_data(df)

    logger.info("Engineering customer features.")

    df = engineer_features(df)

    # Separate target variable
    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Remove target and customer ID
    X = df.drop(
        columns=["Churn", "customerID"]
    )

    if y.isna().any():
        raise ValueError("Invalid target values detected.")

    if len(X) != len(y):
        raise ValueError("Feature and target lengths differ.")

    logger.info("Feature shape: %s", X.shape)

    logger.info("Target distribution:\n%s", y.value_counts())

    return X, y


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

def split_data(X, y):

    logger.info("Splitting dataset into training and testing.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info("Training records: %s", len(X_train))

    logger.info("Testing records: %s", len(X_test))

    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# LOGISTIC REGRESSION MODEL
# --------------------------------------------------

def train_logistic_regression(X_train, y_train):

    logger.info("Building Logistic Regression pipeline.")

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),

            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    logger.info("Training Logistic Regression model.")

    model.fit(X_train, y_train)

    logger.info("Logistic Regression training completed.")

    return model


# --------------------------------------------------
# RANDOM FOREST MODEL
# --------------------------------------------------

def train_random_forest(X_train, y_train):

    logger.info("Building Random Forest pipeline.")

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )
            )
        ]
    )

    logger.info("Training Random Forest model.")

    model.fit(X_train, y_train)

    logger.info("Random Forest training completed.")

    return model


# --------------------------------------------------
# GRADIENT BOOSTING MODEL
# --------------------------------------------------

def train_gradient_boosting(X_train, y_train):

    logger.info("Building Gradient Boosting pipeline.")

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_train)),

            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    logger.info("Training Gradient Boosting model.")

    model.fit(X_train, y_train)

    logger.info("Gradient Boosting training completed.")

    return model


# --------------------------------------------------
# MODEL EVALUATION
# --------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name):

    logger.info("Evaluating %s.", model_name)

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,

        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(y_test, probabilities)
        ),

        "average_precision": float(
            average_precision_score(
                y_test,
                probabilities
            )
        )
    }

    print("\n" + "=" * 60)

    print(f"{model_name.upper()} EVALUATION")

    print("=" * 60)

    for metric, value in metrics.items():

        print(f"{metric}: {value}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

    return metrics


# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

def save_model(model, metrics, model_name):

    model_filename = f"{model_name}_pipeline.joblib"

    metrics_filename = f"{model_name}_metrics.json"

    model_path = ARTIFACT_PATH / model_filename

    metrics_path = ARTIFACT_PATH / metrics_filename

    joblib.dump(model, model_path)

    with open(metrics_path, "w", encoding="utf-8") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    logger.info("Model saved: %s", model_path)

    logger.info("Metrics saved: %s", metrics_path)


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nCUSTOMER CHURN MODEL TRAINING")

    # Prepare data
    X, y = prepare_training_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ==============================================
    # MODEL 1: LOGISTIC REGRESSION
    # ==============================================

    logistic_model = train_logistic_regression(
        X_train,
        y_train
    )

    logistic_metrics = evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    save_model(
        logistic_model,
        logistic_metrics,
        "logistic_regression"
    )

    # ==============================================
    # MODEL 2: RANDOM FOREST
    # ==============================================

    random_forest_model = train_random_forest(
        X_train,
        y_train
    )

    random_forest_metrics = evaluate_model(
        random_forest_model,
        X_test,
        y_test,
        "Random Forest"
    )

    save_model(
        random_forest_model,
        random_forest_metrics,
        "random_forest"
    )

    # ==============================================
    # MODEL 3: GRADIENT BOOSTING
    # ==============================================

    gradient_boosting_model = train_gradient_boosting(
        X_train,
        y_train
    )

    gradient_boosting_metrics = evaluate_model(
        gradient_boosting_model,
        X_test,
        y_test,
        "Gradient Boosting"
    )

    save_model(
        gradient_boosting_model,
        gradient_boosting_metrics,
        "gradient_boosting"
    )

    # ==============================================
    # TRAINING COMPLETED
    # ==============================================

    print("\n" + "=" * 60)

    print("ALL THREE MODELS TRAINED SUCCESSFULLY")

    print("=" * 60)


if __name__ == "__main__":

    main()