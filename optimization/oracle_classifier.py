import numpy as np
import scipy.optimize

from optimization.functions import oracle_risk, oracle_risk_derivative
from optimization.__base_pu_classifier import BasePUClassifier


class OracleClassifier(BasePUClassifier):
    def fit(self, X, y):
        b_init = np.random.random(X.shape[1] + 1) / 100

        res = scipy.optimize.minimize(
            fun=oracle_risk,
            x0=b_init,
            method='BFGS',
            args=(X, y),
            jac=oracle_risk_derivative
        )

        self.params = res.x

        return self
