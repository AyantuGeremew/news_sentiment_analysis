import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_publication_data(df, date_column):
    """
    Convert publication date column to datetime
    and aggregate article counts over time.

    Parameters:
        df (pd.DataFrame): Input dataframe
        date_column (str): Name of publication date column

    Returns:
        pd.DataFrame: Aggregated publication frequency
    """

    # Convert to datetime
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')

    # Remove invalid dates
    df = df.dropna(subset=[date_column])

    # Extract date only
    df['publication_day'] = df[date_column].dt.date

    # Count articles per day
    publication_frequency = (
        df.groupby('publication_day')
        .size()
        .reset_index(name='article_count')
    )

    return publication_frequency


# =========================
# Visualization Function
# =========================
def plot_publication_frequency(publication_frequency):
    """
    Plot article publication frequency over time.

    Parameters:
        publication_frequency (pd.DataFrame): Aggregated frequency dataframe
    """

    plt.figure(figsize=(14, 6))

    plt.plot(
        publication_frequency['publication_day'],
        publication_frequency['article_count'],
        marker='o'
    )

    plt.title('Article Publication Frequency Over Time')
    plt.xlabel('Publication Date')
    plt.ylabel('Number of Articles')

    plt.xticks(rotation=45)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# ==========================================
# Prepare Data + Detect Spikes
# ==========================================
def analyze_publication_spikes(df, date_column, threshold_multiplier=2):
    """
    Prepare data, aggregate frequency, and detect spikes.
    """

    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    df = df.dropna(subset=[date_column])

    df['day'] = df[date_column].dt.date

    freq = df.groupby('day').size().reset_index(name='count')

    mean = freq['count'].mean()
    std = freq['count'].std()
    threshold = mean + threshold_multiplier * std

    spikes = freq[freq['count'] > threshold]

    return freq, spikes, threshold


# ==========================================
# Map Events to Spikes
# ==========================================
def attach_market_events(spikes):
    """
    Map spike dates to known market events.
    """

    events = {
        '2024-01-10': 'Bitcoin ETF Approval',
        '2024-03-14': 'Fed Interest Rate Decision',
        '2024-04-20': 'Tech Earnings Surge',
        '2024-06-12': 'Inflation Data Release',
        '2024-08-05': 'Global Market Selloff',
    }

    spikes['event'] = spikes['day'].astype(str).map(events).fillna("No event")

    return spikes


# ==========================================
# Plot Results
# ==========================================
def plot(freq, spikes, threshold):
    """
    Visualize publication frequency and spikes.
    """

    plt.figure(figsize=(14, 6))

    plt.plot(freq['day'], freq['count'], marker='o', label='Articles')

    plt.scatter(
        spikes['day'],
        spikes['count'],
        color='red',
        s=100,
        label='Spikes'
    )

    plt.axhline(threshold, linestyle='--', label='Threshold')

    plt.title("Publication Frequency with Spike Detection")
    plt.xlabel("Date")
    plt.ylabel("Article Count")

    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
