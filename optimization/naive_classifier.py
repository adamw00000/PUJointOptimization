import time
import numpy as np
import scipy.optimize

from optimization.__base_pu_classifier import BasePUClassifier
from optimization.c_estimation import BaseCEstimator
from optimization.functions import oracle_risk, oracle_risk_derivative, predict_proba, add_bias


class NaiveClassifier(BasePUClassifier):
    c_estimator: BaseCEstimator
    include_bias: bool

    def __init__(self, c_estimator: BaseCEstimator, include_bias: bool = True):
        self.c_estimator = c_estimator
        self.include_bias = include_bias

    def fit(self, X, s, c: float = None):
        t = time.time()
        if self.include_bias:
            X = add_bias(X)
        self.P_S_1 = float(np.mean(s == 1))
        # b_init = np.random.random(X.shape[1]) / 100
        b_init = np.zeros(X.shape[1])

        if c is None:
            self.c_estimate = self.c_estimator.fit(X, s)
        else:
            self.c_estimate = c

        res = scipy.optimize.minimize(
            fun=oracle_risk,
            x0=b_init,
            method='BFGS',
            args=(X, s),
            jac=oracle_risk_derivative
        )
        self.evaluations = res.nfev + res.njev

        self.params = res.x
        self.total_time = time.time() - t

        return self

    def predict_proba(self, X):
        s_proba = predict_proba(X, self.params)
        y_proba = s_proba / self.c_estimate
        return y_proba
