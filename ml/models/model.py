from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd

class Model(ABC):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        self.model: Any = None
        self.input_shape: int = input_shape
        self.output_shape: int = output_shape

    @abstractmethod
    def build(self, **params: Any) -> None:
        pass

    @abstractmethod
    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        pass