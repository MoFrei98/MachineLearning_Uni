import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from typing import Tuple

class Dataset:
    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self._x_train: pd.DataFrame | None = None
        self._y_train: pd.Series | None = None
        self._x_test: pd.DataFrame | None = None
        self._y_test: pd.Series | None = None
        self.test_size: float = test_size
        self.random_state: int = random_state
        self.label_encoder: LabelEncoder = LabelEncoder()

    def load_data(self, dataset_name: str = 'iris') -> 'Dataset':
        ds: pd.DataFrame = sns.load_dataset(dataset_name)
        x: pd.DataFrame = ds.drop('species', axis=1)
        y: pd.Series = ds['species']

        # Encode string labels to numeric values
        y_encoded: np.ndarray = self.label_encoder.fit_transform(y)
        y = pd.Series(y_encoded, index=y.index)

        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        return self

    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._x_train is None:
            self.load_data()
        return self._x_train, self._y_train

    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self._x_test is None:
            self.load_data()
        return self._x_test, self._y_test

    def split_data(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        # ...existing code...
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )