import numpy as np
import scipy.optimize

from optimization.__base_pu_classifier import BasePUClassifier
from optimization.functions import oracle_risk, oracle_risk_derivative, predict_proba


class NaiveClassifier(BasePUClassifier):
    c: float

    def __init__(self, c: float) -> None:
        self.c = c

    def fit(self, X, s):
        b_init = np.random.random(X.shape[1] + 1) / 100

        res = scipy.optimize.minimize(
            fun=oracle_risk,
            x0=b_init,
            method='BFGS',
            args=(X, s),
            jac=oracle_risk_derivative
        )

        self.params = res.x

        return self

    def predict_proba(self, X):
        s_proba = predict_proba(X, self.params)
        y_proba = s_proba / self.c
        return y_proba
