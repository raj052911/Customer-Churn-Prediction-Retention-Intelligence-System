import pandas as pd


def validate_dataset(df):

    print("\n" + "=" * 55)
    print("       CUSTOMER CHURN DATASET VALIDATION")
    print("=" * 55)

    # 1. Dataset information
    print("\n1. DATASET INFORMATION")
    print("-" * 40)

    print("Total Rows:", df.shape[0])
    print("Total Columns:", df.shape[1])

    print("\nColumn Data Types:")
    print(df.dtypes)

    # 2. Missing values
    print("\n2. MISSING VALUE ANALYSIS")
    print("-" * 40)

    missing = df.isnull().sum()

    print(missing[missing > 0])

    print(
        "\nTotal Missing Values:",
        df.isnull().sum().sum()
    )

    # 3. Empty strings
    print("\n3. EMPTY STRING ANALYSIS")
    print("-" * 40)

    object_columns = df.select_dtypes(
        include=["object"]
    )

    empty_values = object_columns.apply(
        lambda col: col.str.strip().eq("").sum()
    )

    print(empty_values[empty_values > 0])

    # 4. Duplicate records
    print("\n4. DUPLICATE ANALYSIS")
    print("-" * 40)

    print(
        "Duplicate Rows:",
        df.duplicated().sum()
    )

    if "customerID" in df.columns:

        print(
            "Duplicate Customer IDs:",
            df["customerID"].duplicated().sum()
        )

    # 5. Target variable
    print("\n5. TARGET VARIABLE ANALYSIS")
    print("-" * 40)

    if "Churn" in df.columns:

        print("\nChurn Distribution:")

        print(
            df["Churn"].value_counts(dropna=False)
        )

        print("\nChurn Percentage:")

        print(
            df["Churn"].value_counts(
                normalize=True,
                dropna=False
            ) * 100
        )

    # 6. TotalCharges validation
    print("\n6. NUMERICAL DATA VALIDATION")
    print("-" * 40)

    if "TotalCharges" in df.columns:

        numeric_values = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

        invalid_values = (
            numeric_values.isna()
            & df["TotalCharges"].notna()
        )

        print(
            "Values requiring inspection:",
            invalid_values.sum()
        )

    print("\n" + "=" * 55)
    print("       VALIDATION COMPLETED")
    print("=" * 55)


if __name__ == "__main__":

    from src.data.load_data import load_dataset

    df = load_dataset()

    validate_dataset(df)