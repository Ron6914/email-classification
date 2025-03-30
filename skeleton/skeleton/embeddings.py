import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from Config import Config
import pickle


class TextEmbedder:
    def __init__(self, method='tfidf', dim_reduction=None):
        """
        Initialize the text embedder

        Args:
            method: 'tfidf' (only TF-IDF supported in this basic version)
            dim_reduction: None or int (number of dimensions for SVD reduction)
        """
        self.method = method
        self.dim_reduction = dim_reduction
        self.vectorizer = None
        self.svd = None

    def fit_transform(self, texts):
        """
        Fit the vectorizer and transform texts to embeddings

        Args:
            texts: List or pandas Series of text documents

        Returns:
            numpy array of embeddings
        """
        self.vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            stop_words='english'
        )
        embeddings = self.vectorizer.fit_transform(texts).toarray()

        if self.dim_reduction:
            self.svd = TruncatedSVD(n_components=self.dim_reduction)
            embeddings = self.svd.fit_transform(embeddings)

        return embeddings

    def transform(self, texts):
        """
        Transform new texts using fitted vectorizer

        Args:
            texts: List or pandas Series of text documents

        Returns:
            numpy array of embeddings
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")

        embeddings = self.vectorizer.transform(texts).toarray()

        if self.dim_reduction and self.svd:
            embeddings = self.svd.transform(embeddings)

        return embeddings

    def save(self, filepath):
        """Save the embedder to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'dim_reduction': self.dim_reduction,
                'vectorizer': self.vectorizer,
                'svd': self.svd
            }, f)

    @classmethod
    def load(cls, filepath):
        """Load embedder from disk"""
        with open(filepath, 'rb') as f:
            params = pickle.load(f)

        embedder = cls(method=params['method'], dim_reduction=params['dim_reduction'])
        embedder.vectorizer = params['vectorizer']
        embedder.svd = params['svd']
        return embedder