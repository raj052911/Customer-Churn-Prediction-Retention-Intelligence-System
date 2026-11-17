from src.data.load_data import load_dataset
from src.preprocessing.preprocessing import clean_customer_data


def main():

    print("\nCUSTOMER DATA CLEANING TEST")

    # Load original dataset
    original_data = load_dataset()

    # Clean a copy
    cleaned_data = clean_customer_data(original_data)

    print("\nOriginal Dataset Shape:")
    print(original_data.shape)

    print("\nCleaned Dataset Shape:")
    print(cleaned_data.shape)

    print("\nMissing Values After Cleaning:")
    print(cleaned_data.isnull().sum())

    print("\nDuplicate Customer IDs:")
    print(cleaned_data["customerID"].duplicated().sum())

    print("\nTotalCharges Data Type:")
    print(cleaned_data["TotalCharges"].dtype)

    print("\nCleaning completed successfully.")


if __name__ == "__main__":
    main()