from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ml.models.model import Model
from typing import Any
import numpy as np
import pandas as pd

class LDAModel(Model):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)
        self.is_trained: bool = False

    def build(self, **params: Any) -> None:
        self.model = LinearDiscriminantAnalysis(**params)

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained using 'fit()' before prediction.")
        return self.model.predict(x)