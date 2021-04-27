import numpy as np
import numpy.typing as npt
import scipy.optimize

from abc import abstractmethod
from optimization.__base_pu_classifier import BasePUClassifier
from optimization.c_estimation.pure_alpha_estimator import PureAlphaEstimator
from optimization.functions import cccp_risk_wrt_c, cccp_risk_derivative_wrt_c


class SplitOptimizationPUClassifier(BasePUClassifier):
    inner_method_name: str
    max_iter: int
    max_inner_iter: int
    verbosity: int

    def __init__(self, inner_method_name: str, tol: float, max_iter: int, max_inner_iter: int,
                 verbosity: int):
        self.inner_method_name = inner_method_name
        self.tol = tol
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.verbosity = verbosity

    @abstractmethod
    def _minimize_wrt_b(self, X, s, c_estimate, old_b_estimate) -> npt.ArrayLike:
        pass

    def _minimize_wrt_c(self, X, s, b_estimate, old_c_estimate) -> float:
        self.P_S_1 = np.mean(s == 1)
        c_estimate = (self.P_S_1 + 1) / 2

        res = scipy.optimize.minimize(
            fun=cccp_risk_wrt_c,
            jac=cccp_risk_derivative_wrt_c,
            x0=np.array([c_estimate]),
            args=(X, s, b_estimate),
            method='TNC',  # better?
            # method='L-BFGS-B',  # faster?
            bounds=[(self.P_S_1, 0.99999)],
            options={
                # 'disp': True
            }
        )

        if self.verbosity > 0:
            print('Estimated c:', res.x[0])

        return res.x[0]

    def fit(self, X, s, c: float = None):
        # b_estimate = np.random.random(X.shape[1] + 1) / 100
        b_estimate = np.zeros(X.shape[1] + 1)

        if c is None:
            P_S_eq_1 = np.mean(s == 1)
            c_estimate = (1 + P_S_eq_1) / 2

            if self.verbosity > 1:
                print('Initial b value:', b_estimate)
                print('Initial c value:', c_estimate)

            for i in range(self.max_iter):
                if self.verbosity > 0:
                    print('Step:', f'{i + 1}/{self.max_iter}')

                new_c_estimate = self._minimize_wrt_c(X, s, b_estimate, c_estimate)

                if i > 0 and np.abs(new_c_estimate - c_estimate) < self.tol:
                    if self.verbosity > 0:
                        print('Procedure converged, stopping...')
                    break

                c_estimate = new_c_estimate
                b_estimate = self._minimize_wrt_b(X, s, c_estimate, b_estimate)

            b_estimate = self._minimize_wrt_b(X, s, c_estimate, b_estimate)

            self.params = b_estimate
            self.c_estimate = c_estimate
        else:
            self.params = self._minimize_wrt_b(X, s, c, b_estimate)
            self.c_estimate = c
