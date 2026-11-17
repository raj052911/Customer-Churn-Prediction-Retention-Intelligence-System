import json
import joblib

from pathlib import Path

from src.utils.logger import get_logger


logger = get_logger(__name__)


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_PATH = PROJECT_ROOT / "artifacts"

BEST_MODEL_PATH = (
    ARTIFACT_PATH / "best_model.joblib"
)

BEST_MODEL_INFO_PATH = (
    ARTIFACT_PATH / "best_model_info.json"
)


# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------

def select_best_model():

    logger.info("Starting final model selection.")

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

    # Select the model with the highest test ROC-AUC.
    # Average Precision is used as a tie-breaker.
    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["test_roc_auc"],
            results[name]["test_average_precision"]
        )
    )

    best_model_file = (
        ARTIFACT_PATH /
        f"{best_model_name}_tuned_pipeline.joblib"
    )

    if not best_model_file.exists():

        raise FileNotFoundError(
            f"Tuned model not found: {best_model_file}"
        )

    # Load the selected model
    best_model = joblib.load(best_model_file)

    # Save final model
    joblib.dump(
        best_model,
        BEST_MODEL_PATH
    )

    # Save selection information
    model_information = {
        "selected_model": best_model_name,
        "selection_metric": "test_roc_auc",
        "test_roc_auc": results[best_model_name]["test_roc_auc"],
        "test_average_precision": results[best_model_name]["test_average_precision"]
    }

    with open(
        BEST_MODEL_INFO_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            model_information,
            file,
            indent=4
        )

    logger.info(
        "Best model selected: %s",
        best_model_name
    )

    logger.info(
        "Final model saved: %s",
        BEST_MODEL_PATH
    )

    print("\n" + "=" * 60)

    print("FINAL MODEL SELECTION")

    print("=" * 60)

    print(f"Selected Model: {best_model_name}")

    print(
        f"Test ROC-AUC: "
        f"{model_information['test_roc_auc']:.4f}"
    )

    print(
        f"Test Average Precision: "
        f"{model_information['test_average_precision']:.4f}"
    )

    print(f"\nSaved Model: {BEST_MODEL_PATH}")

    print("\nFinal model selection completed successfully.")


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    select_best_model()