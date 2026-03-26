from ml.models.model import Model
from ml.dataset import Dataset
from ml.metrics.metric import Metric
from typing import List, Dict, Any

class Trainer:
    def __init__(self, model: Model, dataset: Dataset, metrics_list: List[Metric] | None = None, epochs: int = 1) -> None:
        self.model: Model = model
        self.dataset: Dataset = dataset
        self.metrics_list: List[Metric] = metrics_list if metrics_list is not None else []
        self.epochs: int = epochs

    def train(self) -> None:
        x_train, y_train = self.dataset.get_train_data()
        # Delegation an das Modell
        self.model.fit(x_train, y_train)
        print("Training completed.")

    def evaluate(self) -> Dict[str, Any]:
        x_test, y_test = self.dataset.get_test_data()
        predictions = self.model.predict(x_test)

        results: Dict[str, Any] = {}
        for m in self.metrics_list:
            # Polymorphie: Jede Metrik berechnet sich selbst
            results[m.name] = m.calculate(y_test, predictions)

        return results
