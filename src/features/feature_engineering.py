import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional customer features
    for churn prediction.
    """

    logger.info("Starting feature engineering.")

    data = df.copy()

    # -----------------------------------------
    # 1. TENURE GROUP
    # -----------------------------------------

    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "New",
            "Short_Term",
            "Medium_Term",
            "Long_Term"
        ]
    )

    # -----------------------------------------
    # 2. TOTAL SERVICES
    # -----------------------------------------

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    data["TotalServices"] = (
        data[service_columns]
        .eq("Yes")
        .sum(axis=1)
    )

    # -----------------------------------------
    # 3. AVERAGE MONTHLY SPEND
    # -----------------------------------------

    data["AverageMonthlySpend"] = (
        data["TotalCharges"] /
        data["tenure"].replace(0, float("nan"))
    )

    # -----------------------------------------
    # 4. LONG TERM CUSTOMER
    # -----------------------------------------

    data["IsLongTermCustomer"] = (
        data["tenure"] >= 24
    ).astype(int)

    # -----------------------------------------
    # 5. MONTH-TO-MONTH CONTRACT
    # -----------------------------------------

    data["HasMonthToMonthContract"] = (
        data["Contract"] == "Month-to-month"
    ).astype(int)

    logger.info(
        "Feature engineering completed successfully."
    )

    logger.info(
        "Total features after engineering: %s",
        data.shape[1]
    )

    return data


if __name__ == "__main__":

    from src.data.load_data import load_dataset
    from src.preprocessing.preprocessing import clean_customer_data

    df = load_dataset()

    cleaned_data = clean_customer_data(df)

    engineered_data = engineer_features(cleaned_data)

    print("\nENGINEERED DATASET PREVIEW")

    print(engineered_data.head())

    print("\nNEW FEATURES")

    print([
        "TenureGroup",
        "TotalServices",
        "AverageMonthlySpend",
        "IsLongTermCustomer",
        "HasMonthToMonthContract"
    ])

    print("\nFinal Dataset Shape:")
    print(engineered_data.shape)