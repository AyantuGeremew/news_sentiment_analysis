import pandas as pd
import numpy as np
import pynance as pn
import yfinance as yf

def ensure_correct_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure stock market columns are correctly typed.

    Converts:
    - Open, High, Low, Close -> float
    - Volume -> integer

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected column types.
    """

    # Float columns
    price_columns = ["Open", "High", "Low", "Close"]

    for col in price_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    # Integer column
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(
            df["Volume"], errors="coerce"
        ).astype("Int64")

    return df

def handle_missing_values(df: pd.DataFrame,strategy: str = "drop") -> pd.DataFrame:
    """
    Check for and handle missing values in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    strategy : str, optional
        Method to handle missing values:
        - "drop"  : Remove rows with missing values
        - "mean"  : Fill numeric columns with mean
        - "median": Fill numeric columns with median
        - "ffill" : Forward fill missing values
        - "bfill" : Backward fill missing values

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe.
    """

    # Check missing values
    print("Missing Values Per Column:")
    print(df.isnull().sum())

    # Handle missing values
    if strategy == "drop":
        df = df.dropna()

    elif strategy == "mean":
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].mean()
        )

    elif strategy == "median":
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].median()
        )

    elif strategy == "ffill":
        df = df.ffill()

    elif strategy == "bfill":
        df = df.bfill()

    else:
        raise ValueError(
            "Invalid strategy. Choose from "
            "['drop', 'mean', 'median', 'ffill', 'bfill']"
        )

    # Verify remaining missing values
    print("\nRemaining Missing Values:")
    print(df.isnull().sum())

    return df

def calculate_moving_averages(df: pd.DataFrame,column: str = "Close",windows: list = [5, 10, 20]) -> pd.DataFrame:
    """
    Calculate Simple Moving Average (SMA)
    and Exponential Moving Average (EMA)
    for multiple window sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    column : str
        Column used for calculations.

    windows : list
        List of moving average window sizes.

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA and EMA columns added.
    """

    for window in windows:

        # Simple Moving Average
        df[f"SMA_{window}"] = (
            df[column]
            .rolling(window=window)
            .mean()
        )

        # Exponential Moving Average
        df[f"EMA_{window}"] = (
            df[column]
            .ewm(span=window, adjust=False)
            .mean()
        )

    return df

def calculate_rsi(df, column="close", window=14):
    """
    Calculate RSI and classify market conditions.
    """

    # clean column names
    df.columns = df.columns.str.strip()

    # map lowercase columns
    col_map = {c.lower(): c for c in df.columns}

    if column.lower() not in col_map:
        raise KeyError(f"Column '{column}' not found. Available: {list(df.columns)}")

    col = col_map[column.lower()]

    # price change
    delta = df[col].diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    gain = pd.Series(gain, index=df.index)
    loss = pd.Series(loss, index=df.index)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    df["RSI_Signal"] = df["RSI"].apply(
        lambda x: "Overbought" if x > 70 else "Oversold" if x < 30 else "Neutral"
    )

    return df

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculates MACD, Signal Line, and Histogram.
    """
    # Calculate EMAs
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD Line
    macd_line = ema_fast - ema_slow
    
    # Calculate Signal Line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate Histogram
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'MACD': macd_line,
        'Signal': signal_line,
        'Histogram': histogram
    })

def get_macd_signals(macd_df):
    """
    Detects buy/sell signals based on MACD/Signal line crossovers.
    """
    signals = pd.DataFrame(index=macd_df.index)
    signals['Buy_Signal'] = (macd_df['MACD'] > macd_df['Signal']) & \
                            (macd_df['MACD'].shift(1) <= macd_df['Signal'].shift(1))
    
    signals['Sell_Signal'] = (macd_df['MACD'] < macd_df['Signal']) & \
                             (macd_df['MACD'].shift(1) >= macd_df['Signal'].shift(1))
    
    return signals
