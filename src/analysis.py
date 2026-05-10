import pandas as pd
import matplotlib.pyplot as plt

def get_length_stats(data, column_name):
    """
    Calculates character counts and returns descriptive statistics.
    """
    lengths = data[column_name].str.len()
    
    stats = {
        'mean': lengths.mean(),
        'median': lengths.median(),
        'std': lengths.std(),
        'min': lengths.min(),
        'max': lengths.max(),
        'raw_lengths': lengths
    }
    return stats

def plot_length_distribution(lengths, title="Text Length Distribution", color="skyblue"):
    """
    Handles the visualization of the distribution.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=30, color=color, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Character Count')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def print_summary(stats, label="Text"):
    """
    Prints a clean summary of the statistics.
    """
    print(f"--- Statistics for {label} ---")
    print(f"Average Length: {stats['mean']:.2f}")
    print(f"Median Length:  {stats['median']}")
    print(f"Std Deviation:  {stats['std']:.2f}")
    print(f"Range:         {stats['min']} - {stats['max']}")
    print("-" * 30)

def get_publisher_counts(data, column_name='publisher'):
    """
    Returns the count of articles for each publisher, sorted by activity.
    """
    # value_counts() handles the counting and sorting (descending) automatically
    counts = data[column_name].value_counts()
    return counts

def plot_publisher_activity(counts, top_n=10):
    """
    Visualizes the top N most active publishers.
    """
    top_publishers = counts.head(top_n)
    
    plt.figure(figsize=(12, 6))
    top_publishers.plot(kind='bar', color='coral', edgecolor='black')
    
    plt.title(f'Top {top_n} Most Active Publishers')
    plt.xlabel('Publisher Name')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def print_activity_report(counts):
    """
    Prints a formatted list of the top active sources.
    """
    print(f"{'PUBLISHER':<25} | {'ARTICLES':<10}")
    print("-" * 40)
    for publisher, count in counts.items():
        print(f"{publisher:<25} | {count:<10}")

