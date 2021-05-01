import time
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

    c_function_evals: int = 0
    c_jacobian_evals: int = 0
    b_function_evals: int = 0
    b_jacobian_evals: int = 0

    def __init__(self, inner_method_name: str, tol: float, max_iter: int, max_inner_iter: int,
                 verbosity: int):
        self.inner_method_name = inner_method_name
        self.tol = tol
        self.max_iter = max_iter
        self.max_inner_iter = max_inner_iter
        self.verbosity = verbosity

    @abstractmethod
    def _minimize_wrt_b(self, X, s, c_estimate, old_b_estimate) -> (npt.ArrayLike, int, int):
        pass

    def _minimize_wrt_c(self, X, s, b_estimate, old_c_estimate) -> float:
        c_estimate = (self.P_S_1 + 1) / 2

        res = scipy.optimize.minimize(
            fun=cccp_risk_wrt_c,
            jac=cccp_risk_derivative_wrt_c,
            x0=np.array([c_estimate]),
            args=(X, s, b_estimate),
            # method='TNC',  # better?
            method='L-BFGS-B',  # faster?
            bounds=[(self.P_S_1, 0.99999)],
            options={
                # 'disp': True
            }
        )

        self.c_function_evals += res.nfev
        self.c_jacobian_evals += getattr(res, 'njev', 0)

        if self.verbosity > 0:
            print('Estimated c:', res.x[0])

        return res.x[0]

    def fit(self, X, s, c: float = None):
        self.iterations = 0

        self.c_function_evals = 0
        self.c_jacobian_evals = 0
        self.b_function_evals = 0
        self.b_jacobian_evals = 0

        t = time.time()
        self.P_S_1 = np.mean(s == 1)
        # b_estimate = np.random.random(X.shape[1] + 1) / 100
        b_estimate = np.zeros(X.shape[1] + 1)

        if c is None:
            c_estimate = (1 + self.P_S_1) / 2

            if self.verbosity > 1:
                print('Initial b value:', b_estimate)
                print('Initial c value:', c_estimate)

            for i in range(self.max_iter):
                self.iterations += 1
                if self.verbosity > 0:
                    print('Step:', f'{self.iterations}/{self.max_iter}')

                new_c_estimate = self._minimize_wrt_c(X, s, b_estimate, c_estimate)

                if i > 0 and np.abs(new_c_estimate - c_estimate) < self.tol:
                    if self.verbosity > 0:
                        print('Procedure converged, stopping...')
                    break

                c_estimate = new_c_estimate
                b_estimate, n_fevals, n_jevals = self._minimize_wrt_b(X, s, c_estimate, b_estimate)
                self.b_function_evals += n_fevals
                self.b_jacobian_evals += n_jevals

            b_estimate, n_fevals, n_jevals = self._minimize_wrt_b(X, s, c_estimate, b_estimate)
            self.b_function_evals += n_fevals
            self.b_jacobian_evals += n_jevals

            self.params = b_estimate
            self.c_estimate = c_estimate
        else:
            self.params, n_fevals, n_jevals = self._minimize_wrt_b(X, s, c, b_estimate)
            self.c_estimate = c

            self.iterations = 1
            self.b_function_evals += n_fevals
            self.b_jacobian_evals += n_jevals

        self.total_time = time.time() - t
        self.evaluations = self.c_function_evals + self.c_jacobian_evals + self.b_function_evals + self.b_jacobian_evals
