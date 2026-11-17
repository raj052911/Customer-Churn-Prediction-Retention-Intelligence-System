import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    roc_auc_score
)

from src.data.load_data import load_dataset

from src.preprocessing.preprocessing import clean_customer_data

from src.features.feature_engineering import engineer_features

from src.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_PATH = PROJECT_ROOT / "artifacts"

REPORT_PATH = PROJECT_ROOT / "reports"

FIGURE_PATH = REPORT_PATH / "figures"

FIGURE_PATH.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = ARTIFACT_PATH / "best_model.joblib"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

def prepare_test_data():

    logger.info("Loading customer dataset.")

    df = load_dataset()

    df = clean_customer_data(df)

    df = engineer_features(df)

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop(
        columns=["Churn", "customerID"]
    )

    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_test, y_test


# --------------------------------------------------
# LOAD SELECTED MODEL
# --------------------------------------------------

def load_best_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Best model not found."
        )

    logger.info("Loading selected model.")

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# GENERATE PREDICTIONS
# --------------------------------------------------

def generate_predictions(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    return predictions, probabilities


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

def generate_confusion_matrix(y_test, predictions):

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["No Churn", "Churn"]
    )

    display.plot()

    plt.title("Customer Churn Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    return matrix


# --------------------------------------------------
# ROC CURVE
# --------------------------------------------------

def generate_roc_curve(y_test, probabilities):

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"ROC-AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("Customer Churn ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "roc_curve.png",
        dpi=300
    )

    plt.close()

    return float(roc_auc)


# --------------------------------------------------
# PRECISION RECALL CURVE
# --------------------------------------------------

def generate_precision_recall_curve(
    y_test,
    probabilities
):

    precision, recall, thresholds = precision_recall_curve(
        y_test,
        probabilities
    )

    average_precision = average_precision_score(
        y_test,
        probabilities
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"Average Precision = {average_precision:.4f}"
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Customer Churn Precision-Recall Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_PATH / "precision_recall_curve.png",
        dpi=300
    )

    plt.close()

    return float(average_precision)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

def generate_classification_report(
    y_test,
    predictions
):

    report = classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"],
        output_dict=True,
        zero_division=0
    )

    report_path = (
        REPORT_PATH /
        "classification_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print("\nCLASSIFICATION REPORT")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Churn", "Churn"],
            zero_division=0
        )
    )

    return report


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nFINAL CUSTOMER CHURN MODEL EVALUATION")

    X_test, y_test = prepare_test_data()

    model = load_best_model()

    predictions, probabilities = generate_predictions(
        model,
        X_test,
        y_test
    )

    matrix = generate_confusion_matrix(
        y_test,
        predictions
    )

    roc_auc = generate_roc_curve(
        y_test,
        probabilities
    )

    average_precision = generate_precision_recall_curve(
        y_test,
        probabilities
    )

    generate_classification_report(
        y_test,
        predictions
    )

    print("\nCONFUSION MATRIX")

    print(matrix)

    print(f"\nROC-AUC: {roc_auc:.4f}")

    print(
        f"Average Precision: {average_precision:.4f}"
    )

    print("\n" + "=" * 60)

    print("FINAL MODEL EVALUATION COMPLETED")

    print("=" * 60)


if __name__ == "__main__":

    main()