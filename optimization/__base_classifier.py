import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod
from optimization.functions import predict_proba


class BaseClassifier(ABC):
    params: npt.ArrayLike

    def set_params(self, params: npt.ArrayLike) -> None:
        self.params = params

    def get_params(self) -> npt.ArrayLike:
        return self.params

    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> 'BaseClassifier':
        pass

    def predict_proba(self, X: npt.ArrayLike) -> npt.ArrayLike:
        y_proba = predict_proba(X, self.params)
        return y_proba

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        y_proba = self.predict_proba(X)
        y_pred = np.where(y_proba > 0.5, 1, 0)
        return y_pred
