import numpy as np
import scipy.optimize

from optimization.functions import cccp_risk_wrt_b, cccp_risk_wrt_c, cccp_risk_derivative_wrt_b, cccp_risk_derivative_wrt_c
from optimization.__base_pu_classifier import BasePUClassifier


class CccpClassifier(BasePUClassifier):
    tol: float
    max_iter: int
    cccp_max_iter: int
    cg_max_iter: int
    verbosity: int

    c_estimate: float

    def __init__(self, tol: float = 1e-4, max_iter: int = 50, cccp_max_iter: int = 10, cg_max_iter: int = 100,
                 verbosity: int = 0):
        self.tol = tol
        self.max_iter = max_iter
        self.cccp_max_iter = cccp_max_iter
        self.cg_max_iter = cg_max_iter
        self.verbosity = verbosity

    def fit(self, X, s):
        b_estimate = np.random.random(X.shape[1] + 1) / 100
        c_estimate = 0.5

        if self.verbosity > 1:
            print('Initial b value:', b_estimate)
            print('Initial c value:', c_estimate)

        for i in range(self.max_iter):
            if self.verbosity > 0:
                print('Step:', f'{i + 1}/{self.max_iter}')
            res = scipy.optimize.minimize(
                fun=cccp_risk_wrt_c,
                jac=cccp_risk_derivative_wrt_c,
                x0=c_estimate,
                args=(X, s, b_estimate),
                method='TNC',
                bounds=[(0, 1)]
            )

            if self.verbosity > 0:
                print('Estimated c:', res.x[0])

            if i > 0 and np.abs(res.x - c_estimate) < self.tol:
                if self.verbosity > 0:
                    print('Procedure converged, stopping...')
                break

            c_estimate = res.x[0]

            for j in range(self.cccp_max_iter):
                if self.verbosity > 1:
                    print('CCCP step:', f'{j + 1}/{self.max_iter}')
                res = scipy.optimize.minimize(
                    fun=cccp_risk_wrt_b,
                    jac=cccp_risk_derivative_wrt_b,
                    x0=b_estimate,
                    args=(X, s, c_estimate, b_estimate),
                    method='CG',
                    options={
                        'maxiter': self.cg_max_iter
                    }
                )

                if self.verbosity > 1:
                    print('Estimated b:', res.x)

                if j > 0 and np.max(np.abs(res.x - b_estimate)) < self.tol:
                    if self.verbosity > 1:
                        print('CCCP converged, stopping...')
                    break
                b_estimate = res.x

        self.params = b_estimate
        self.c_estimate = c_estimate
