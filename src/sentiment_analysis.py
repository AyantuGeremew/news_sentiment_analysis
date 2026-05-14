import pandas as pd
import numpy as np
import nltk 
import pandas_market_calendars as mcal 
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer 

def normalize_timestamps(df, time_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)
    df["trading_day"] = df[time_col].dt.date
    return df


def adjust_news_to_trading_day(df, time_col, cutoff_hour=16):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', utc=True)

    hour = df[time_col].dt.hour

    df["trading_day"] = np.where(
        hour >= cutoff_hour,
        (df[time_col] + pd.Timedelta(days=1)).dt.date,
        df[time_col].dt.date
    )

    return df


def align_on_trading_day(news_df, stock_df):
    return pd.merge(news_df, stock_df, on="trading_day", how="inner")

def align_news_to_trading_days(news_df, time_col="timestamp",
                              market="NYSE",
                              start_date=None,
                              end_date=None):
    """
    Align news timestamps to next valid trading day (handles weekends + holidays).
    """

    df = news_df.copy()

    # Convert to datetime
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Build trading calendar
    if start_date is None:
        start_date = df[time_col].min()
    if end_date is None:
        end_date = df[time_col].max()

    calendar = mcal.get_calendar(market)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    trading_days = set(schedule.index.normalize())

    # Function to shift to next trading day
    def get_next_trading_day(date):
        date = pd.Timestamp(date).normalize()

        if date in trading_days:
            return date

        next_day = date + pd.Timedelta(days=1)
        while next_day not in trading_days:
            next_day += pd.Timedelta(days=1)

        return next_day

    # Apply mapping
    df["trading_day"] = df[time_col].apply(get_next_trading_day)

    return df

# Download VADER lexicon (run once)
nltk.download("vader_lexicon")

def align_news_to_trading_day(news_df,
                              time_col="date",
                              market="NYSE"):

    df = news_df.copy()

    # Convert safely
    df[time_col] = pd.to_datetime(
        df[time_col],
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=[time_col])

    # Create wider calendar range
    start_date = df[time_col].min() - pd.Timedelta(days=10)
    end_date = df[time_col].max() + pd.Timedelta(days=10)

    # Trading calendar
    calendar = mcal.get_calendar(market)

    schedule = calendar.schedule(
        start_date=start_date,
        end_date=end_date
    )

    trading_days = sorted(schedule.index.normalize())

    # Function to get next trading day safely
    def get_next_trading_day(date):

        date = pd.Timestamp(date).normalize()

        # Return same day if trading day
        if date in trading_days:
            return date

        # Find first future trading day
        future_days = [d for d in trading_days if d >= date]

        # If no future day exists
        if len(future_days) == 0:
            return pd.NaT

        return future_days[0]

    # Apply safely
    df["trading_day"] = df[time_col].apply(get_next_trading_day)

    return df

def add_sentiment_score(df, text_column="headline"):
    """
    Apply sentiment analysis and add a numerical sentiment score.
    Uses VADER compound score (-1 to +1).
    """

    df = df.copy()

    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Compute sentiment score
    df["sentiment_score"] = df[text_column].apply(
        lambda x: sia.polarity_scores(str(x))["compound"]
    )

    return df

def apply_textblob_sentiment(df, text_column="headline"):
    """
    Apply TextBlob sentiment analysis.
    Returns polarity score between -1 and 1.
    """

    df = df.copy()

    df["textblob_sentiment"] = df[text_column].apply(
        lambda text: TextBlob(str(text)).sentiment.polarity
    )

    return df


def apply_vader_sentiment(df, text_column="headline"):
    """
    Apply VADER sentiment analysis.
    Returns compound score between -1 and 1.
    """

    df = df.copy()

    sia = SentimentIntensityAnalyzer()

    df["vader_sentiment"] = df[text_column].apply(
        lambda text: sia.polarity_scores(str(text))["compound"]
    )

    return df


def apply_all_sentiments(df, text_column="headline"):
    """
    Apply both TextBlob and VADER sentiment analysis.
    """

    df = apply_textblob_sentiment(df, text_column)
    df = apply_vader_sentiment(df, text_column)

    return df