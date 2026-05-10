import pandas as pd
import numpy as np

def analyze_publishers(df, date_column="date", publisher_column="publisher", top_n=10):
    """
    Identify and characterize most active publishers.

    Returns:
        DataFrame of top publishers with activity and behavior metrics.
    """

    # --- Data preparation ---
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column, publisher_column])

    # --- Basic aggregation ---
    activity = (
        df.groupby(publisher_column)
        .agg(
            total_articles=(publisher_column, "count"),
            first_publish=(date_column, "min"),
            last_publish=(date_column, "max"),
            active_days=(date_column, lambda x: x.dt.date.nunique())
        )
        .reset_index()
    )

    # --- Behavioral characterization ---
    behavior = (
        df.groupby(publisher_column)
        .agg(
            avg_articles_per_active_day=(date_column, lambda x: x.count() / x.dt.date.nunique()),
            publishing_variability=(date_column, lambda x: x.value_counts().std() if len(x) > 1 else 0),
        )
        .reset_index()
    )

    # --- Merge metrics ---
    result = activity.merge(behavior, on=publisher_column, how="left")

    # --- Rank and filter top publishers ---
    result = result.sort_values("total_articles", ascending=False).head(top_n)

    return result

def analyze_publisher_domains(df, publisher_column="publisher", top_n=10):
    """
    Extracts domains from email-based publishers and analyzes organizational contribution patterns.

    Returns:
        DataFrame with domain-level activity statistics.
    """

    df = df.copy()

    # --- Step 1: Keep only valid email-like publishers ---
    df = df[df[publisher_column].str.contains("@", na=False)]

    # --- Step 2: Extract domain ---
    df["domain"] = df[publisher_column].str.split("@").str[-1].str.lower()

    # --- Step 3: Aggregate domain-level statistics ---
    domain_stats = (
        df.groupby("domain")
        .agg(
            total_articles=("domain", "count"),
            unique_publishers=(publisher_column, "nunique"),
            active_days=("domain", lambda x: x.size),  # fallback if no date column
        )
        .reset_index()
    )

    # --- Step 4: Add contribution share ---
    total_articles = domain_stats["total_articles"].sum()
    domain_stats["contribution_share"] = domain_stats["total_articles"] / total_articles

    # --- Step 5: Rank domains ---
    domain_stats = domain_stats.sort_values("total_articles", ascending=False).head(top_n)

    return domain_stats