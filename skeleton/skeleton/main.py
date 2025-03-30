from modelling.data_model import Data
from model.hierarchichal import HierarchicalClassifier

from preprocess import Preprocessor
import pandas as pd


def main():
        # 1. Load data
        print("1. Loading data...")
        df = pd.read_csv("data/Purchasing.csv")
        print(f"Data loaded with {len(df)} rows")

        # 2. Preprocess
        print("\n2. Preprocessing...")
        preprocessor = Preprocessor(df)
        df = preprocessor.preprocess_text()
        X = preprocessor.vectorize_text()

        # 3. Prepare splits
        print("\n3. Preparing data splits...")
        data = Data(X, df)
        print(f"Train samples: {data.X_train.shape[0]}, Test samples: {data.X_test.shape[0]}")

        # 4. Train model
        print("\n4. Training hierarchical model...")
        model = HierarchicalClassifier()
        model.train(data)

        # 5. Evaluate
        print("\n5. Evaluating model...")
        model.print_results(data)

        print("\nCOMPLETE!")

        # 1. Load data
        print("1. Loading data...")
        df = pd.read_csv("data/AppGallery.csv")
        print(f"Data loaded with {len(df)} rows")

        # 2. Preprocess
        print("\n2. Preprocessing...")
        preprocessor = Preprocessor(df)
        df = preprocessor.preprocess_text()
        X = preprocessor.vectorize_text()

        # 3. Prepare splits
        print("\n3. Preparing data splits...")
        data = Data(X, df)
        print(f"Train samples: {data.X_train.shape[0]}, Test samples: {data.X_test.shape[0]}")

        # 4. Train model
        print("\n4. Training hierarchical model...")
        model = HierarchicalClassifier()
        model.train(data)

        # 5. Evaluate
        print("\n5. Evaluating model...")
        model.print_results(data)

        print("\nCOMPLETE!")
if __name__ == "__main__":
    main()
