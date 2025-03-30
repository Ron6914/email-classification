import pandas as pd
import numpy as np
from Config import Config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            stop_words='english'
        )
        self.X = None
        self.y = None

    def clean_targets(self):
        """Handle missing values in target columns"""
        print("\nCleaning target columns:")

        # Fill NaN values in each target column
        for i, col in enumerate(Config.TYPE_COLS):
            if col not in self.df.columns:
                raise ValueError(f"Target column '{col}' not found in data")

            na_count = self.df[col].isna().sum()
            if na_count > 0:
                print(f" - {na_count} NaN values found in {col}, filling with '{Config.MISSING_CATEGORY}'")
                self.df[col] = self.df[col].fillna(Config.MISSING_CATEGORY)
            else:
                print(f" - {col} has no missing values")

        # Verify no NaN values remain
        if self.df[Config.TYPE_COLS].isna().any().any():
            raise ValueError("NaN values still present in target columns after cleaning")

        return self.df

    def preprocess_text(self):
        """Combine text columns and clean"""
        # Clean targets first
        self.clean_targets()

        # Handle text columns
        for col in Config.TEXT_COLUMNS:
            self.df[col] = self.df[col].fillna('')

        self.df['combined_text'] = (
                self.df[Config.TICKET_SUMMARY] + " " +
                self.df[Config.INTERACTION_CONTENT]
        )
        return self.df

    def vectorize_text(self):
        """Convert text to TF-IDF vectors"""
        self.X = self.vectorizer.fit_transform(self.df['combined_text'])
        print(f"\nText vectorized to shape: {self.X.shape}")
        return self.X

    def prepare_targets(self):
        """Prepare target variables"""
        self.y = self.df[Config.TYPE_COLS]
        return self.y

    def get_clean_data(self):
        """Return cleaned data"""
        return self.X, self.df

    def split_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        self.train_df = pd.DataFrame(X_train)
        self.test_df = pd.DataFrame(X_test)
        self.train_df[Config.TYPE_COLS] = y_train
        self.test_df[Config.TYPE_COLS] = y_test
        return X_train, X_test, y_train, y_test
#Methods related to data loading and all pre-processing steps will go here