import numpy as np
import numpy.typing as npt
import scipy.optimize

from optimization.functions import cccp_risk_wrt_b, cccp_risk_derivative_wrt_b
from optimization.__split_optimization_pu_classifier import SplitOptimizationPUClassifier


class CccpClassifier(SplitOptimizationPUClassifier):
    cg_max_iter: int

    def __init__(self, tol: float = 1e-4, max_iter: int = 100, cccp_max_iter: int = 1000, cg_max_iter: int = 1000,
                 verbosity: int = 0):
        super().__init__('CCCP', tol=tol, max_iter=max_iter, max_inner_iter=cccp_max_iter, verbosity=verbosity)
        self.cg_max_iter = cg_max_iter

    def _minimize_wrt_b(self, X, s, c_estimate, old_b_estimate) -> (npt.ArrayLike, int, int):
        n_fevals = 0
        n_jevals = 0

        # b_estimate = old_b_estimate
        b_estimate = np.zeros_like(old_b_estimate)

        for j in range(self.max_inner_iter):
            if self.verbosity > 1:
                print(f'{self.inner_method_name} step:', f'{j + 1}/{self.max_inner_iter}')

            res = scipy.optimize.minimize(
                fun=cccp_risk_wrt_b,
                jac=cccp_risk_derivative_wrt_b,
                x0=b_estimate,
                args=(X, s, c_estimate, b_estimate),
                # method='CG',
                method='CG',
                options={
                    'maxiter': self.cg_max_iter,
                    'disp': self.verbosity > 1,
                }
            )
            new_b_estimate = res.x

            n_fevals += res.nfev
            n_jevals += res.njev

            if self.verbosity > 1:
                print('Estimation success:', res.success, f'({res.nit} iterations)')
                if not res.success:
                    print('Solver status:', res.status)
                    print('Message:', res.message)
                print('Estimated b:', new_b_estimate)

            if j > 0 and np.min(np.abs(new_b_estimate - b_estimate)) < self.tol:
                if self.verbosity > 1:
                    print('CCCP converged, stopping...')
                break
            b_estimate = new_b_estimate

        return b_estimate, n_fevals, n_jevals
