import pandas as pd
import matplotlib.pyplot as plt

def plot_macd(price_data, macd_df, signals):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot Price and Signals
    ax1.plot(price_data, label='Price', color='black', alpha=0.3)
    ax1.scatter(signals.index[signals['Buy_Signal']], price_data[signals['Buy_Signal']], 
                marker='^', color='green', label='Buy Signal', s=100)
    ax1.scatter(signals.index[signals['Sell_Signal']], price_data[signals['Sell_Signal']], 
                marker='v', color='red', label='Sell Signal', s=100)
    ax1.set_title('Price & MACD Crossover Signals')
    ax1.legend()

    # Plot MACD Components
    ax2.plot(macd_df['MACD'], label='MACD Line', color='blue')
    ax2.plot(macd_df['Signal'], label='Signal Line', color='orange')
    ax2.bar(macd_df.index, macd_df['Histogram'], label='Histogram', 
           color=['green' if x > 0 else 'red' for x in macd_df['Histogram']])
    ax2.set_title('MACD Oscillator')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def plot_close_with_moving_averages(df: pd.DataFrame,price_column: str = "close",sma_windows: list = [5, 20],
    ema_windows: list = [5, 20],title: str = "Close Price with Moving Averages"):
    """
    Plot closing price with SMA and EMA overlays.

    Parameters
    ----------
    df : pd.DataFrame
        Input stock dataframe.

    price_column : str
        Column used for price (close).

    sma_windows : list
        List of SMA windows.

    ema_windows : list
        List of EMA windows.

    title : str
        Plot title.

    Returns
    -------
    None
    """

    # Clean column names
    df.columns = df.columns.str.strip().str.lower()

    if price_column.lower() not in df.columns:
        raise KeyError(f"Column '{price_column}' not found. Available: {list(df.columns)}")

    price_col = price_column.lower()

    plt.figure(figsize=(12, 6))

    # Plot closing price
    plt.plot(df.index, df[price_col], label="Close Price", linewidth=2)

    # SMA plots
    for window in sma_windows:
        sma = df[price_col].rolling(window=window).mean()
        plt.plot(df.index, sma, label=f"SMA {window}")

    # EMA plots
    for window in ema_windows:
        ema = df[price_col].ewm(span=window, adjust=False).mean()
        plt.plot(df.index, ema, label=f"EMA {window}")

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_rsi_and_macd(df: pd.DataFrame,rsi_column: str = "RSI",macd_column: str = "MACD",
    signal_column: str = "MACD_Signal", histogram_column: str = "MACD_Histogram"):
    """
    Plot RSI and MACD indicators in separate panels.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing RSI and MACD values.

    Returns
    -------
    None
    """

    df.columns = df.columns.str.strip()

    # Create subplots (2 panels)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # -------------------------
    # RSI PLOT (Panel 1)
    # -------------------------
    axes[0].plot(df.index, df[rsi_column], label="RSI", color="blue")
    axes[0].axhline(70, linestyle="--", color="red", label="Overbought (70)")
    axes[0].axhline(30, linestyle="--", color="green", label="Oversold (30)")

    axes[0].set_title("Relative Strength Index (RSI)")
    axes[0].set_ylabel("RSI Value")
    axes[0].legend()
    axes[0].grid(True)

    # -------------------------
    # MACD PLOT (Panel 2)
    # -------------------------
    axes[1].plot(df.index, df[macd_column], label="MACD", color="blue")
    axes[1].plot(df.index, df[signal_column], label="Signal Line", color="red")

    # Histogram as bar chart
    axes[1].bar(df.index, df[histogram_column], label="Histogram", color="gray", alpha=0.3)

    axes[1].set_title("MACD Indicator")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True)

    # Layout
    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()

def plot_price_with_indicators(
    df: pd.DataFrame,
    price_column: str = "close",
    sma_windows: list = [20],
    ema_windows: list = [20],
    rsi_column: str = "RSI",
    macd_column: str = "MACD",
    signal_column: str = "MACD_Signal"
):
    """
    Visualize price action and how indicators relate to it.

    Panels:
    1. Price + Moving Averages
    2. RSI
    3. MACD
    """

    df.columns = df.columns.str.strip().str.lower()

    price_col = price_column.lower()

    if price_col not in df.columns:
        raise KeyError(f"Column '{price_column}' not found: {list(df.columns)}")

    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # ---------------------------
    # 1. PRICE + MOVING AVERAGES
    # ---------------------------
    axes[0].plot(df.index, df[price_col], label="Close Price", linewidth=2)

    for w in sma_windows:
        sma = df[price_col].rolling(w).mean()
        axes[0].plot(df.index, sma, label=f"SMA {w}")

    for w in ema_windows:
        ema = df[price_col].ewm(span=w, adjust=False).mean()
        axes[0].plot(df.index, ema, label=f"EMA {w}")

    axes[0].set_title("Price Action with Moving Averages")
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True)

    # ---------------------------
    # 2. RSI
    # ---------------------------
    if rsi_column in df.columns:
        axes[1].plot(df.index, df[rsi_column], label="RSI", color="blue")
        axes[1].axhline(70, linestyle="--", color="red")
        axes[1].axhline(30, linestyle="--", color="green")

        axes[1].set_title("RSI (Momentum)")
        axes[1].set_ylabel("RSI")
        axes[1].legend()
        axes[1].grid(True)

    # ---------------------------
    # 3. MACD
    # ---------------------------
    if macd_column in df.columns and signal_column in df.columns:
        axes[2].plot(df.index, df[macd_column], label="MACD", color="blue")
        axes[2].plot(df.index, df[signal_column], label="Signal", color="red")

        axes[2].bar(
            df.index,
            df[macd_column] - df[signal_column],
            label="Histogram",
            alpha=0.3
        )

        axes[2].set_title("MACD (Trend Shifts)")
        axes[2].set_ylabel("MACD")
        axes[2].legend()
        axes[2].grid(True)

    plt.xlabel("Time")
    plt.tight_layout()
    plt.show()