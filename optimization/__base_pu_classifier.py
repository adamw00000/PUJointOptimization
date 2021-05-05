import numpy as np

from abc import ABC, abstractmethod

from optimization.functions import predict_proba
from optimization.c_estimation.pure_alpha_estimator import PureAlphaEstimator


class BasePUClassifier(ABC):
    _P_S_1: float
    _c_estimate: float
    _params: np.array
    _total_time: float
    _evaluations: int
    _iterations: int = 1

    def set_params(self, params: np.array) -> None:
        self.params = params

    def get_params(self) -> np.array:
        return self.params

    @abstractmethod
    def fit(self, X, s, c: float):
        pass

    def predict_proba(self, X):
        y_proba = predict_proba(X, self.params)
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.where(y_proba > 0.5, 1, 0)
        return y_pred

    def get_STD_alpha(self):
        alpha_estimator = PureAlphaEstimator(self.c_estimate, self.P_S_1)
        return alpha_estimator.get_STD_alpha()

    def get_CC_alpha(self):
        alpha_estimator = PureAlphaEstimator(self.c_estimate, self.P_S_1)
        return alpha_estimator.get_CC_alpha()

    @property
    def c_estimate(self) -> float:
        return self._c_estimate

    @c_estimate.setter
    def c_estimate(self, value: float):
        self._c_estimate = value

    @property
    def P_S_1(self) -> float:
        return self._P_S_1

    @P_S_1.setter
    def P_S_1(self, value: float):
        self._P_S_1 = value

    @property
    def params(self) -> np.array:
        return self._params

    @params.setter
    def params(self, value: np.array):
        self._params = value

    @property
    def total_time(self) -> float:
        return self._total_time

    @total_time.setter
    def total_time(self, value: float):
        self._total_time = value

    @property
    def evaluations(self) -> int:
        return self._evaluations

    @evaluations.setter
    def evaluations(self, value: int):
        self._evaluations = value

    @property
    def iterations(self) -> int:
        return self._iterations

    @iterations.setter
    def iterations(self, value: int):
        self._iterations = value




