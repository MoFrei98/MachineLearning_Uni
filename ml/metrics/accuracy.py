from sklearn.metrics import accuracy_score
from ml.metrics.metric import Metric
from typing import Callable
import numpy as np
import pandas as pd

class Accuracy(Metric):
    def __init__(self, metric_function: Callable = accuracy_score) -> None:
        super().__init__(name="Accuracy")
        self.metric_function: Callable = metric_function

    def calculate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> float:
        return self.metric_function(y_true, y_pred)