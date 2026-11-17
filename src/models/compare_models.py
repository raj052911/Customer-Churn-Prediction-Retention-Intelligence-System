import json

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_PATH = PROJECT_ROOT / "artifacts"

REPORT_PATH = PROJECT_ROOT / "reports"

REPORT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD TUNED MODEL RESULTS
# --------------------------------------------------

def load_model_results():

    comparison_file = (
        ARTIFACT_PATH /
        "tuned_model_comparison.json"
    )

    if not comparison_file.exists():

        raise FileNotFoundError(
            "Tuned model comparison file not found."
        )

    with open(
        comparison_file,
        "r",
        encoding="utf-8"
    ) as file:

        results = json.load(file)

    logger.info("Model comparison results loaded.")

    return results


# --------------------------------------------------
# LOAD DETAILED MODEL METRICS
# --------------------------------------------------

def load_detailed_metrics():

    model_names = [
        "logistic_regression",
        "random_forest",
        "gradient_boosting"
    ]

    records = []

    for model_name in model_names:

        metrics_file = (
            ARTIFACT_PATH /
            f"{model_name}_tuned_metrics.json"
        )

        if not metrics_file.exists():

            raise FileNotFoundError(
                f"Metrics file not found: {metrics_file}"
            )

        with open(
            metrics_file,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        records.append({

            "model": model_name,

            "accuracy": data["test_metrics"]["accuracy"],

            "precision": data["test_metrics"]["precision"],

            "recall": data["test_metrics"]["recall"],

            "f1_score": data["test_metrics"]["f1_score"],

            "roc_auc": data["test_metrics"]["roc_auc"],

            "average_precision": data["test_metrics"]["average_precision"],

            "best_cv_roc_auc": data["best_cv_roc_auc"]

        })

    return pd.DataFrame(records)


# --------------------------------------------------
# COMPARE MODELS
# --------------------------------------------------

def compare_models():

    logger.info("Starting model comparison.")

    comparison_data = load_model_results()

    detailed_results = load_detailed_metrics()

    print("\n" + "=" * 75)

    print("CUSTOMER CHURN MODEL COMPARISON")

    print("=" * 75)

    print(
        detailed_results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # Rank using test ROC-AUC for initial comparison
    ranked_results = detailed_results.sort_values(
        by="roc_auc",
        ascending=False
    )

    best_model = ranked_results.iloc[0]["model"]

    print("\n" + "=" * 75)

    print("TOP MODEL BY TEST ROC-AUC")

    print("=" * 75)

    print(f"Selected candidate: {best_model}")

    print(
        f"Test ROC-AUC: "
        f"{ranked_results.iloc[0]['roc_auc']:.4f}"
    )

    print(
        f"Test Average Precision: "
        f"{ranked_results.iloc[0]['average_precision']:.4f}"
    )

    # Save comparison report
    report_file = (
        REPORT_PATH /
        "model_comparison.csv"
    )

    detailed_results.to_csv(
        report_file,
        index=False
    )

    logger.info(
        "Model comparison report saved: %s",
        report_file
    )

    return detailed_results, best_model


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    compare_models()

    print("\nModel comparison completed successfully.")


if __name__ == "__main__":

    main()