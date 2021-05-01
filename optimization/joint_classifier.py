import time
import numpy as np
import scipy.optimize
import typing

from optimization.functions import joint_risk, joint_risk_derivative
from optimization.__base_pu_classifier import BasePUClassifier


class JointClassifier(BasePUClassifier):
    def fit(self, X, s, c: float = None):
        t = time.time()
        self.P_S_1 = np.mean(s == 1)

        if c is None:
            c_init = (1 + self.P_S_1) / 2

            # b_init = np.random.random(X.shape[1] + 2) / 100
            b_init = np.zeros(X.shape[1] + 2)
            b_init[-1] = c_init  # initial c

            bounds_type = typing.List[typing.Tuple[typing.Union[float, None], typing.Union[float, None]]]

            bounds: bounds_type = [(None, None) for _ in range(X.shape[1] + 1)]
            bounds.append((self.P_S_1, 0.99999))

            res = scipy.optimize.minimize(
                fun=joint_risk,
                x0=b_init,
                method='L-BFGS-B',
                args=(X, s),
                jac=joint_risk_derivative,
                bounds=bounds
            )
            function_evals = res.nfev
            jacobian_evals = res.njev

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
            function_evals = res.nfev
            jacobian_evals = res.njev

            self.params = res.x
            self.c_estimate = c

        self.total_time = time.time() - t
        self.evaluations = function_evals + jacobian_evals
