from sklearn.svm import SVC
from ml.models.model import Model
from ml.kernel.kernel import Kernel
from typing import Any
import numpy as np
import pandas as pd

class SVMModel(Model):
    def __init__(self, input_shape: int, output_shape: int) -> None:
        super().__init__(input_shape, output_shape)
        self.model: SVC | None = None
        self.is_trained: bool = False

    def build(self, kernel: Kernel | str | None = None, **params: Any) -> None:
        if kernel is None or isinstance(kernel, str):
            # Fallback for string kernels
            self.model = SVC(kernel=kernel or 'rbf', **params)
        elif isinstance(kernel, Kernel):
            # Use Kernel object and merge its params with additional params
            kernel_params: dict = kernel.get_params()
            kernel_params.update(params)
            self.model = SVC(**kernel_params)
        else:
            raise TypeError("kernel must be a Kernel object or a string")

    def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model must be trained using 'fit()' before prediction.")
        return self.model.predict(x)