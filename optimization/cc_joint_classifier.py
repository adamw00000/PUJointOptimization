import time
import numpy as np
import scipy.optimize
import typing

from optimization.functions import joint_risk, joint_risk_derivative, joint_risk_with_info, \
    joint_risk_derivative_with_info, cc_joint_risk, cc_joint_risk_derivative, add_bias
from optimization.__base_pu_classifier import BasePUClassifier


class CcJointClassifier(BasePUClassifier):
    tol: float
    max_iter: int
    alpha_estimate: float
    include_bias: bool

    def __init__(self, tol: float = 1e-4, max_iter: int = 1000, include_bias: bool = True):
        self.tol = tol
        self.max_iter = max_iter
        self.include_bias = include_bias

    def fit(self, X, s, c: float = None):
        t = time.time()
        if self.include_bias:
            X = add_bias(X)
        self.P_S_1 = np.mean(s == 1)

        if c is None:
            alpha_init = 0.5

            # b_init = np.random.random(X.shape[1] + 1) / 100
            b_init = np.zeros(X.shape[1] + 1)
            b_init[-1] = alpha_init  # initial alpha

            bounds_type = typing.List[typing.Tuple[typing.Union[float, None], typing.Union[float, None]]]

            bounds: bounds_type = [(None, None) for _ in range(X.shape[1] + 1)]
            bounds.append((0.00001, 0.99999))

            res = scipy.optimize.minimize(
                fun=cc_joint_risk,
                x0=b_init,
                method='L-BFGS-B',
                args=(X, s, self.P_S_1),
                jac=cc_joint_risk_derivative,
                bounds=bounds,
                options={
                    'maxiter': self.max_iter,
                    'gtol': self.tol,
                }
            )
            function_evals = res.nfev
            jacobian_evals = res.njev

            self.params = res.x[:-1]
            self.alpha_estimate = res.x[-1]
            self.c_estimate = self.P_S_1 / (self.alpha_estimate * (1 - self.P_S_1) + self.P_S_1)
        else:
            self.alpha_estimate = ((1 - c) * self.P_S_1) / (c * (1 - self.P_S_1))
            self.c_estimate = c
            b_init = np.random.random(X.shape[1] + 1) / 100

            res = scipy.optimize.minimize(
                fun=cc_joint_risk,
                x0=b_init,
                method='BFGS',
                args=(X, s, self.P_S_1, self.alpha_estimate),
                jac=cc_joint_risk_derivative,
                options={
                    'maxiter': self.max_iter,
                    'gtol': self.tol,
                }
            )
            function_evals = res.nfev
            jacobian_evals = res.njev

            self.params = res.x

        self.total_time = time.time() - t
        self.evaluations = function_evals + jacobian_evals

    def get_CC_alpha(self):
        return self.alpha_estimate
