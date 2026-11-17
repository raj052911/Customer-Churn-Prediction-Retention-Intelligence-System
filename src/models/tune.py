import json
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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
CV_FOLDS = 5
N_ITER = 10


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def prepare_data():

    logger.info("Loading customer churn dataset.")

    df = load_dataset()

    logger.info("Cleaning customer data.")

    df = clean_customer_data(df)

    logger.info("Engineering customer features.")

    df = engineer_features(df)

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop(
        columns=["Churn", "customerID"]
    )

    if y.isna().any():
        raise ValueError("Invalid target values detected.")

    return X, y


# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------

def split_data(X, y):

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
# CROSS VALIDATION
# --------------------------------------------------

def create_cross_validation():

    return StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE
    )


# --------------------------------------------------
# MODEL CONFIGURATIONS
# --------------------------------------------------

def get_model_configurations(X_train):

    preprocessor = build_preprocessor(X_train)

    configurations = {

        "logistic_regression": {

            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),

                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=3000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE
                        )
                    )
                ]
            ),

            "parameters": {
                "classifier__C": [
                    0.01,
                    0.1,
                    1,
                    10,
                    100
                ],

                "classifier__solver": [
                    "liblinear",
                    "lbfgs"
                ]
            }
        },

        "random_forest": {

            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),

                    (
                        "classifier",
                        RandomForestClassifier(
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                            n_jobs=-1
                        )
                    )
                ]
            ),

            "parameters": {
                "classifier__n_estimators": [
                    100,
                    200,
                    300,
                    500
                ],

                "classifier__max_depth": [
                    None,
                    10,
                    20,
                    30
                ],

                "classifier__min_samples_split": [
                    2,
                    5,
                    10
                ],

                "classifier__min_samples_leaf": [
                    1,
                    2,
                    4
                ],

                "classifier__max_features": [
                    "sqrt",
                    "log2"
                ]
            }
        },

        "gradient_boosting": {

            "pipeline": Pipeline(
                steps=[
                    ("preprocessor", preprocessor),

                    (
                        "classifier",
                        GradientBoostingClassifier(
                            random_state=RANDOM_STATE
                        )
                    )
                ]
            ),

            "parameters": {
                "classifier__n_estimators": [
                    100,
                    150,
                    200,
                    300
                ],

                "classifier__learning_rate": [
                    0.01,
                    0.05,
                    0.1,
                    0.2
                ],

                "classifier__max_depth": [
                    2,
                    3,
                    4,
                    5
                ],

                "classifier__subsample": [
                    0.8,
                    0.9,
                    1.0
                ]
            }
        }
    }

    return configurations


# --------------------------------------------------
# HYPERPARAMETER TUNING
# --------------------------------------------------

def tune_model(
    model_name,
    pipeline,
    parameters,
    X_train,
    y_train,
    cv
):

    logger.info("Starting tuning for %s.", model_name)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameters,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        refit=True,
        return_train_score=True
    )

    search.fit(
        X_train,
        y_train
    )

    logger.info("Tuning completed for %s.", model_name)

    logger.info(
        "Best CV ROC-AUC for %s: %.4f",
        model_name,
        search.best_score_
    )

    logger.info(
        "Best parameters for %s: %s",
        model_name,
        search.best_params_
    )

    return search


# --------------------------------------------------
# FINAL TEST EVALUATION
# --------------------------------------------------

def evaluate_tuned_model(
    model_name,
    best_model,
    X_test,
    y_test
):

    predictions = best_model.predict(X_test)

    probabilities = best_model.predict_proba(X_test)[:, 1]

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

    print(f"{model_name.upper()} - TUNED MODEL EVALUATION")

    print("=" * 60)

    for metric, value in metrics.items():

        print(f"{metric}: {value}")

    return metrics


# --------------------------------------------------
# SAVE TUNED MODEL
# --------------------------------------------------

def save_tuned_model(
    model_name,
    best_model,
    best_parameters,
    best_cv_score,
    test_metrics
):

    model_path = (
        ARTIFACT_PATH /
        f"{model_name}_tuned_pipeline.joblib"
    )

    metrics_path = (
        ARTIFACT_PATH /
        f"{model_name}_tuned_metrics.json"
    )

    joblib.dump(
        best_model,
        model_path
    )

    results = {

        "model": model_name,

        "best_cv_roc_auc": float(best_cv_score),

        "best_parameters": best_parameters,

        "test_metrics": test_metrics
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    logger.info("Tuned model saved: %s", model_path)

    logger.info("Tuned metrics saved: %s", metrics_path)


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nCUSTOMER CHURN HYPERPARAMETER TUNING")

    # Prepare data
    X, y = prepare_data()

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Cross-validation configuration
    cv = create_cross_validation()

    # Model configurations
    configurations = get_model_configurations(X_train)

    all_results = {}

    # Tune each model
    for model_name, configuration in configurations.items():

        print("\n" + "=" * 60)

        print(f"TUNING: {model_name.upper()}")

        print("=" * 60)

        search = tune_model(
            model_name,
            configuration["pipeline"],
            configuration["parameters"],
            X_train,
            y_train,
            cv
        )

        # Evaluate best estimator on untouched test data
        test_metrics = evaluate_tuned_model(
            model_name,
            search.best_estimator_,
            X_test,
            y_test
        )

        # Save tuned model and metrics
        save_tuned_model(
            model_name,
            search.best_estimator_,
            search.best_params_,
            search.best_score_,
            test_metrics
        )

        all_results[model_name] = {
            "best_cv_roc_auc": float(search.best_score_),
            "test_roc_auc": test_metrics["roc_auc"],
            "test_average_precision": test_metrics["average_precision"]
        }

    # Compare tuned models
    print("\n" + "=" * 60)

    print("TUNED MODEL COMPARISON")

    print("=" * 60)

    for model_name, results in all_results.items():

        print(f"\nModel: {model_name}")

        for metric, value in results.items():

            print(f"{metric}: {value}")

    # Save comparison results
    comparison_path = (
        ARTIFACT_PATH /
        "tuned_model_comparison.json"
    )

    with open(
        comparison_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4
        )

    print("\n" + "=" * 60)

    print("HYPERPARAMETER TUNING COMPLETED")

    print("=" * 60)


if __name__ == "__main__":

    main()