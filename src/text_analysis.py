import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def extract_top_phrases(corpus, n_gram_range=(1, 1), top_n=10, stopwords='english'):
    """
    Extracts the most frequent words or phrases.
    n_gram_range=(1, 1) for single words, (2, 2) for bigrams, etc.
    """
    # Initialize the Vectorizer
    vec = CountVectorizer(
        ngram_range=n_gram_range, 
        stop_words=stopwords
    ).fit(corpus)
    
    # Sum up the occurrences of each word
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    
    # Create a list of (word, frequency) tuples
    words_freq = [
        (word, sum_words[0, idx]) 
        for word, idx in vec.vocabulary_.items()
    ]
    
    # Sort by frequency descending
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    
    return words_freq[:top_n]

def format_as_dataframe(freq_list, columns=['Phrase', 'Count']):
    """Converts the list of tuples into a clean DataFrame."""
    return pd.DataFrame(freq_list, columns=columns)


def extract_recurring_themes(corpus, min_freq=2, top_n=20):

    # Clean text
    corpus = corpus.dropna().astype(str)

    # Vectorizer
    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=(2, 2),
        min_df=min_freq,
        max_features=5000
    )

    # Sparse matrix
    X = vectorizer.fit_transform(corpus)

    # Efficient sparse computation
    frequencies = X.sum(axis=0).A1

    # Get phrases
    phrases = vectorizer.get_feature_names_out()

    # Create dataframe
    themes_df = pd.DataFrame({
        'theme': phrases,
        'frequency': frequencies
    })

    # Sort
    themes_df = themes_df.sort_values(
        by='frequency',
        ascending=False
    )

    return themes_df.head(top_n)
    
def get_top_counts(corpus, n_gram_range=(1,1), top_n=10):

    if isinstance(corpus, pd.DataFrame):
        corpus = corpus.iloc[:, 0]

    corpus = corpus.dropna().astype(str)
    corpus = corpus[corpus.str.strip() != ""]

    vectorizer = CountVectorizer(
        stop_words='english',
        ngram_range=n_gram_range
    )

    X = vectorizer.fit_transform(corpus)

    vocab = vectorizer.get_feature_names_out()

    # ✅ SAFE: sum sparse matrix directly
    counts = np.asarray(X.sum(axis=0)).flatten()

    top_indices = counts.argsort()[::-1][:top_n]

    return [(vocab[i], counts[i]) for i in top_indices]

def get_top_tfidf(corpus, n_gram_range=(1,1), top_n=10):

    # ensure clean text
    corpus = [str(doc) for doc in corpus if str(doc).strip() != ""]

    if len(corpus) == 0:
        raise ValueError("Corpus is empty after cleaning.")

    vectorizer = TfidfVectorizer(
        ngram_range=n_gram_range,
        stop_words=None  # temporarily disable for debugging
    )

    X = vectorizer.fit_transform(corpus)

    if len(vectorizer.get_feature_names_out()) == 0:
        raise ValueError("No vocabulary found. Check preprocessing or stopwords.")

    feature_names = vectorizer.get_feature_names_out()

    scores = X.sum(axis=0).A1
    top_indices = scores.argsort()[::-1][:top_n]

    return [(feature_names[i], scores[i]) for i in top_indices]

def get_lda_topics(corpus, n_topics=3, words_per_topic=5):
    """
    Groups headlines into abstract topics using LDA.
    Best for: 'What are the broad categories of news?'
    """
    vec = CountVectorizer(stop_words='english')
    matrix = vec.fit_transform(corpus)
    
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(matrix)
    
    words = vec.get_feature_names_out()
    topic_list = []
    
    for i, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[-words_per_topic:]
        keywords = [words[j] for j in top_indices]
        topic_list.append({"Topic_ID": i, "Keywords": ", ".join(keywords)})
        
    return pd.DataFrame(topic_list)