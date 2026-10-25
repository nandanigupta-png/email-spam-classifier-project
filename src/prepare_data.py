"""Step 1: Load and prepare the dataset (Appendix A of the report)."""
import pandas as pd


def load_and_prepare(path="data/spam_dataset.csv"):
    """Read the CSV, clean it, and return a DataFrame with 'message' and 'label'."""
    # A.2 - Loading the dataset
    df = pd.read_csv(path)
    print("First 5 rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # A.3 - Preparing the data
    df = df[["message", "label"]].dropna()
    df["message"] = df["message"].astype(str)
    df["label"] = df["label"].str.lower()
    df["label"] = df["label"].replace({"ham": "legitimate"})
    return df


if __name__ == "__main__":
    data = load_and_prepare()
    print("\nPrepared data:")
    print(data.head())
    print("\nTotal messages:", len(data))
