import pandas as pd


def load_data(filepath):

    # Load CSV
    df = pd.read_csv(filepath)

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    # Convert date column safely
    if 'date' in df.columns:

        df['date'] = pd.to_datetime(
            df['date'],
            format='mixed',
            errors='coerce',
            utc=True
        )

    return df

def remove_nulls(df):
    """Remove rows that have any missing values. Print a summary of what was removed."""
    before = len(df)
    df = df.dropna()
    after = len(df)
    removed = before - after
    if removed > 0:
        print(f"Removed {removed} row(s) with missing values. {after} rows remaining.")
    else:
        print(f"No missing values found. All {after} rows kept.")
    return df
