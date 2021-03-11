import numpy as np
import scipy.optimize

from optimization.functions import joint_risk, joint_risk_derivative
from optimization.__base_pu_classifier import BasePUClassifier


class JointClassifier(BasePUClassifier):
    c_estimate: float

    def fit(self, X, s):
        b_init = np.random.random(X.shape[1] + 2) / 100

        bounds = [(None, None) for i in range(X.shape[1] + 1)]
        bounds.append((0, 1))

        res = scipy.optimize.minimize(
            fun=joint_risk,
            x0=b_init,
            method='L-BFGS-B',
            args=(X, s),
            jac=joint_risk_derivative,
            bounds=bounds
        )

        self.params = res.x[:-1]
        self.c_estimate = res.x[-1]