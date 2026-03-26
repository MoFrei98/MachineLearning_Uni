from sklearn.metrics import confusion_matrix
from ml.metrics.metric import Metric
import pandas as pd
import numpy as np

class ConfusionMatrix(Metric):
    def __init__(self) -> None:
        super().__init__(name="Confusion Matrix")

    def calculate(self, y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> str:
        labels: list = sorted(y_true.unique())
        cm: np.ndarray = confusion_matrix(y_true, y_pred)

        summary: list[str] = []
        for i, label in enumerate(labels):
            # Korrekte sind auf der Diagonale (i == j)
            correct: int = cm[i][i]
            # Fehler sind alle in der Zeile, außer dem korrekten Wert
            total_actual: int = sum(cm[i])
            errors: int = total_actual - correct

            line: str = f"{label:12} -> Corret: {correct:2} | Incorrect: {errors:2}"
            summary.append(line)

        return "\n".join(summary)