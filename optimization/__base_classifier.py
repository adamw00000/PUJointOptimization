import numpy as np

from abc import ABC, abstractmethod
from optimization.functions import predict_proba


class BaseClassifier(ABC):
    params: np.array

    def set_params(self, params: np.array) -> None:
        self.params = params

    def get_params(self) -> np.array:
        return self.params

    @abstractmethod
    def fit(self, X: np.array, y: np.array) -> 'BaseClassifier':
        pass

    def predict_proba(self, X: np.array) -> np.array:
        y_proba = predict_proba(X, self.params)
        return y_proba

    def predict(self, X: np.array) -> np.array:
        y_proba = self.predict_proba(X)
        y_pred = np.where(y_proba > 0.5, 1, 0)
        return y_pred
