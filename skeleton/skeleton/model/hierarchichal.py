from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from model.base import BaseModel
import numpy as np
from Config import Config
from sklearn.exceptions import UndefinedMetricWarning
import warnings

class HierarchicalClassifier(BaseModel):
    def __init__(self):
        super().__init__()  # Initialize BaseModel
        self.models = {
            'Type2': RandomForestClassifier(n_estimators=1000, random_state=Config.RANDOM_SEED),
            'Type3': {},
            'Type4': {}
        }

    def train(self, data):
        """Train all levels of the hierarchy"""
        X_train = data.X_train.toarray() if hasattr(data.X_train, 'toarray') else data.X_train

        # Level 1 - Type2
        self.models['Type2'].fit(X_train, data.Type2_train)

        # Level 2 - Type3
        for type2 in np.unique(data.Type2_train):
            mask = (data.Type2_train == type2).values
            if sum(mask) > 1:  # Need at least 2 samples
                self.models['Type3'][type2] = RandomForestClassifier(
                    n_estimators=500,
                    random_state=Config.RANDOM_SEED
                )
                self.models['Type3'][type2].fit(X_train[mask], data.Type3_train[mask])

                # Level 3 - Type4
                self.models['Type4'][type2] = {}
                for type3 in np.unique(data.Type3_train[mask]):
                    type3_mask = mask & (data.Type3_train == type3).values
                    if sum(type3_mask) > 1:
                        self.models['Type4'][type2][type3] = RandomForestClassifier(
                            n_estimators=300,
                            random_state=Config.RANDOM_SEED
                        )
                        self.models['Type4'][type2][type3].fit(
                            X_train[type3_mask],
                            data.Type4_train[type3_mask]
                        )

    def predict(self, X_test):
        """Predict all hierarchy levels"""
        X_test = X_test.toarray() if hasattr(X_test, 'toarray') else X_test
        preds = {'Type2': [], 'Type3': [], 'Type4': []}

        for i in range(X_test.shape[0]):
            # Level 1 (Type2)
            type2_pred = self.models['Type2'].predict(X_test[i:i+1])[0]
            preds['Type2'].append(type2_pred)

            # Level 2 (Type3)
            type3_pred = None
            if type2_pred in self.models['Type3']:
                type3_pred = self.models['Type3'][type2_pred].predict(X_test[i:i+1])[0]
            preds['Type3'].append(type3_pred)

            # Level 3 (Type4)
            type4_pred = None
            if (type3_pred and
                type2_pred in self.models['Type4'] and
                type3_pred in self.models['Type4'][type2_pred]):
                type4_pred = self.models['Type4'][type2_pred][type3_pred].predict(X_test[i:i+1])[0]
            preds['Type4'].append(type4_pred)

        return preds

    def print_results(self, data):
        """Print classification reports for all levels"""
        warnings.filterwarnings('ignore', category=UndefinedMetricWarning)
        preds = self.predict(data.X_test)

        print("\n" + "=" * 50)
        print("HIERARCHICAL CLASSIFICATION RESULTS")
        print("=" * 50)

        # Level 2 Report
        print("\nLEVEL 2 (Type2) RESULTS:")
        print(classification_report(data.Type2_test, preds['Type2'], zero_division=0))

        # Level 3 Report (filter None predictions)
        valid_type3 = [(p, t) for p, t in zip(preds['Type3'], data.Type3_test) if p is not None]
        if valid_type3:
            print("\nLEVEL 3 (Type3) RESULTS:")
            print(classification_report(
                [t for p, t in valid_type3],
                [p for p, t in valid_type3],
                zero_division=0
            ))

        # Level 4 Report (filter None predictions)
        valid_type4 = [(p, t) for p, t in zip(preds['Type4'], data.Type4_test) if p is not None]
        if valid_type4:
            print("\nLEVEL 4 (Type4) RESULTS:")
            print(classification_report(
                [t for p, t in valid_type4],
                [p for p, t in valid_type4],
                zero_division=0
            ))

    def data_transform(self) -> None:
        pass

    def build(self, values={}):
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self