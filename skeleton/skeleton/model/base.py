from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self) -> None:
        self.defaults = {}

    @abstractmethod
    def train(self) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        pass

    @abstractmethod
    def predict(self) -> int:
        """
        Make predictions using the trained model
        """
        pass

    @abstractmethod
    def data_transform(self) -> None:
        """
        Transform data before training/prediction
        """
        pass

    def build(self, values={}):
        values = values if isinstance(values, dict) else {}
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self