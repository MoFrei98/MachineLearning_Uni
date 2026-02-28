from sklearn.svm import SVC
from ml.models.model import Model

class SVMModel(Model):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        self.model = None
        self.is_trained = False

    def build(self, kernel='rbf', **params):
        self.model = SVC(kernel=kernel, **params)

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x):
        if not self.is_trained:
            raise ValueError("Model must be trained using 'fit()' before prediction.")
        return self.model.predict(x)