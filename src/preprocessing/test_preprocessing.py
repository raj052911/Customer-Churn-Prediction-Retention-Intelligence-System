from src.data.load_data import load_dataset

from src.preprocessing.preprocessing import (
    clean_customer_data,
    build_preprocessor
)

from src.features.feature_engineering import engineer_features


def main():

    print("\nPREPROCESSING PIPELINE TEST")

    # Load dataset
    df = load_dataset()

    # Clean dataset
    df = clean_customer_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Separate features and target
    X = df.drop(
        columns=["Churn", "customerID"]
    )

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(X)

    # Test preprocessing
    X_processed = preprocessor.fit_transform(X)

    # Display results
    print("\nOriginal Feature Shape:")
    print(X.shape)

    print("\nProcessed Feature Shape:")
    print(X_processed.shape)

    print("\nTarget Distribution:")
    print(y.value_counts())

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()