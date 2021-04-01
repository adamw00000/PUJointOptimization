import numpy as np
import scipy.optimize

from optimization.functions import joint_risk, joint_risk_derivative
from optimization.__base_pu_classifier import BasePUClassifier


class JointClassifier(BasePUClassifier):
    c_estimate: float

    def fit(self, X, s, c: float = None):
        if c is None:
            P_S_eq_1 = np.mean(s == 1)
            c_init = (1 + P_S_eq_1) / 2

            # b_init = np.random.random(X.shape[1] + 2) / 100
            b_init = np.zeros(X.shape[1] + 2)
            b_init[-1] = c_init  # initial c

            bounds = [(None, None) for i in range(X.shape[1] + 1)]
            bounds.append((P_S_eq_1, 0.99999))

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
        else:
            b_init = np.random.random(X.shape[1] + 1) / 100

            res = scipy.optimize.minimize(
                fun=joint_risk,
                x0=b_init,
                method='BFGS',
                args=(X, s, c),
                jac=joint_risk_derivative
            )

            self.params = res.x
            self.c_estimate = c
