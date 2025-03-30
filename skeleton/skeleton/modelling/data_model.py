import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import Config
import random

seed = Config.RANDOM_SEED
random.seed(seed)
np.random.seed(seed)


class Data():
    def __init__(self, X: np.ndarray, df: pd.DataFrame) -> None:
        # Verify required columns exist
        missing_cols = [col for col in Config.TYPE_COLS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        self.X = X
        self.df = df

        # Split data for all hierarchical levels
        (self.X_train, self.X_test,
         self.Type2_train, self.Type2_test,
         self.Type3_train, self.Type3_test,
         self.Type4_train, self.Type4_test) = train_test_split(
            X,
            df[Config.TYPE_COLS[0]],  # Type2
            df[Config.TYPE_COLS[1]],  # Type3
            df[Config.TYPE_COLS[2]],  # Type4
            test_size=Config.TEST_SIZE,
            random_state=seed
        )

    # Getters for Type2
    def get_Type2_train(self):
        return self.Type2_train

    def get_Type2_test(self):
        return self.Type2_test

    # Getters for Type3
    def get_Type3_train(self):
        return self.Type3_train

    def get_Type3_test(self):
        return self.Type3_test

    # Getters for Type4
    def get_Type4_train(self):
        return self.Type4_train

    def get_Type4_test(self):
        return self.Type4_test

    # Combined getters
    def get_train_data(self):
        return (self.X_train, self.Type2_train, self.Type3_train, self.Type4_train)

    def get_test_data(self):
        return (self.X_test, self.Type2_test, self.Type3_test, self.Type4_test)

