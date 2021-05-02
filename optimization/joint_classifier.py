import time
import numpy as np
import scipy.optimize
import typing

from optimization.functions import joint_risk, joint_risk_derivative, joint_risk_with_info, \
    joint_risk_derivative_with_info
from optimization.__base_pu_classifier import BasePUClassifier


class JointClassifier(BasePUClassifier):
    tol: float
    max_iter: int
    get_info: bool

    risk_values: typing.List[float]
    param_history: typing.List[typing.List[float]]
    risk_values_no_inner: typing.List[float]

    def __init__(self, tol: float = 1e-4, max_iter: int = 1000, get_info=False):
        self.get_info = get_info
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, s, c: float = None):
        t = time.time()
        self.P_S_1 = np.mean(s == 1)

        self.risk_values = []
        self.param_history = []
        convergence_info = {'param_history': [], 'risk_values': []}

        if c is None:
            c_init = (1 + self.P_S_1) / 2

            # b_init = np.random.random(X.shape[1] + 2) / 100
            b_init = np.zeros(X.shape[1] + 2)
            b_init[-1] = c_init  # initial c

            bounds_type = typing.List[typing.Tuple[typing.Union[float, None], typing.Union[float, None]]]

            bounds: bounds_type = [(None, None) for _ in range(X.shape[1] + 1)]
            bounds.append((self.P_S_1, 0.99999))

            res = scipy.optimize.minimize(
                fun=joint_risk_with_info if self.get_info else joint_risk,
                x0=b_init,
                method='L-BFGS-B',
                args=(X, s, convergence_info) if self.get_info else (X, s),
                jac=joint_risk_derivative_with_info if self.get_info else joint_risk_derivative,
                bounds=bounds,
                options={
                    'maxiter': self.max_iter,
                    'gtol': self.tol,
                }
            )
            function_evals = res.nfev
            jacobian_evals = res.njev

            self.params = res.x[:-1]
            self.c_estimate = res.x[-1]
        else:
            b_init = np.random.random(X.shape[1] + 1) / 100

            res = scipy.optimize.minimize(
                fun=joint_risk_with_info if self.get_info else joint_risk,
                x0=b_init,
                method='BFGS',
                args=(X, s, convergence_info, c) if self.get_info else (X, s, c),
                jac=joint_risk_derivative_with_info if self.get_info else joint_risk_derivative,
                options={
                    'maxiter': self.max_iter,
                    'gtol': self.tol,
                }
            )
            function_evals = res.nfev
            jacobian_evals = res.njev

            self.params = res.x
            self.c_estimate = c

        if self.get_info:
            self.risk_values = convergence_info['risk_values']
            self.param_history = convergence_info['param_history']
            self.risk_values_no_inner = self.risk_values

        self.total_time = time.time() - t
        self.evaluations = function_evals + jacobian_evals
