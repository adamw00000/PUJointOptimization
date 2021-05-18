import numpy as np
import scipy.optimize

from optimization.functions import oracle_risk, oracle_risk_derivative, add_bias
from optimization.__base_classifier import BaseClassifier


class OracleClassifier(BaseClassifier):
    include_bias: bool

    def __init__(self, include_bias: bool = True):
        self.include_bias = include_bias

    def fit(self, X, y):
        if self.include_bias:
            X = add_bias(X)
        # b_init = np.random.random(X.shape[1]) / 100
        b_init = np.zeros(X.shape[1])

        res = scipy.optimize.minimize(
            fun=oracle_risk,
            x0=b_init,
            method='BFGS',
            args=(X, y),
            jac=oracle_risk_derivative
        )

        self.params = res.x

        return self
