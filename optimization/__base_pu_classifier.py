import numpy as np

from optimization.functions import predict_proba


class BasePUClassifier:
    params: np.array

    def set_params(self, params: np.array) -> None:
        self.params = params

    def get_params(self) -> np.array:
        return self.params

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        y_proba = predict_proba(X, self.params)
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.where(y_proba > 0.5, 1, 0)
        return y_pred

