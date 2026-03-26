from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pandas as pd

class Metric(ABC):
    def __init__(self, name: str) -> None:
        self.name: str = name

    @abstractmethod
    def calculate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> Any:
        pass