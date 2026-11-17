import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

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


# --------------------------------------------------
# LOAD SELECTED MODEL
# --------------------------------------------------

def load_best_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Best model not found. Run model selection first."
        )

    logger.info("Loading selected model.")

    model = joblib.load(MODEL_PATH)

    return model


# --------------------------------------------------
# EXTRACT FEATURE IMPORTANCE
# --------------------------------------------------

def extract_feature_importance(model):

    logger.info("Extracting feature importance.")

    # Get preprocessing pipeline
    preprocessor = model.named_steps["preprocessor"]

    # Get classifier
    classifier = model.named_steps["classifier"]

    # Get transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    # Get feature importance values
    importance_values = classifier.feature_importances_

    if len(feature_names) != len(importance_values):

        raise ValueError(
            "Feature names and importance values do not match."
        )

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance": importance_values

    })

    # Sort by importance
    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    logger.info("Feature importance extracted successfully.")

    return importance_df


# --------------------------------------------------
# SAVE FEATURE IMPORTANCE
# --------------------------------------------------

def save_feature_importance(importance_df):

    output_path = (
        REPORT_PATH /
        "feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False
    )

    logger.info(
        "Feature importance saved: %s",
        output_path
    )


# --------------------------------------------------
# VISUALIZE FEATURE IMPORTANCE
# --------------------------------------------------

def visualize_feature_importance(importance_df):

    top_features = importance_df.head(15).sort_values(
        by="importance",
        ascending=True
    )

    plt.figure(figsize=(11, 7))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel("Feature Importance")

    plt.ylabel("Customer Features")

    plt.title(
        "Top 15 Features Influencing Customer Churn"
    )

    plt.tight_layout()

    output_path = (
        FIGURE_PATH /
        "feature_importance.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    logger.info(
        "Feature importance chart saved: %s",
        output_path
    )


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def main():

    print("\nCUSTOMER CHURN MODEL EXPLAINABILITY")

    # Load model
    model = load_best_model()

    # Extract feature importance
    importance_df = extract_feature_importance(
        model
    )

    # Display top features
    print("\nTOP 15 CUSTOMER CHURN FEATURES")

    print(
        importance_df.head(15).to_string(
            index=False
        )
    )

    # Save results
    save_feature_importance(
        importance_df
    )

    # Generate visualization
    visualize_feature_importance(
        importance_df
    )

    print("\n" + "=" * 60)

    print("MODEL EXPLAINABILITY COMPLETED")

    print("=" * 60)


if __name__ == "__main__":

    main()