from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset location
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"


def load_dataset(file_path=DATASET_PATH):
    """
    Load the customer churn dataset.

    Parameters:
        file_path: Path to the CSV dataset.

    Returns:
        pandas.DataFrame
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("The dataset is empty.")

    print("\nDataset loaded successfully!")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst Five Records:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    return df


if __name__ == "__main__":
    load_dataset()