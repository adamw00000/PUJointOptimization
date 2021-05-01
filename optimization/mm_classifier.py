import osqp
import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix

from optimization.__split_optimization_pu_classifier import SplitOptimizationPUClassifier
from optimization.functions import mm_q, add_bias


class MMClassifier(SplitOptimizationPUClassifier):
    cg_max_iter: int

    def __init__(self, tol: float = 1e-4, max_iter: int = 100, mm_max_iter: int = 1000,
                 verbosity: int = 0):
        super().__init__('MM', tol=tol, max_iter=max_iter, max_inner_iter=mm_max_iter, verbosity=verbosity)

    def _minimize_wrt_b(self, X, s, c, old_b_estimate) -> (npt.ArrayLike, int, int):
        n_evals = 0

        # b_estimate = old_b_estimate
        b_estimate = np.zeros_like(old_b_estimate)

        X_with_bias = add_bias(X)
        P = csc_matrix(np.matmul(X_with_bias.T, X_with_bias) / 4)

        for j in range(self.max_inner_iter):
            q = mm_q(b_estimate, X, s, c)

            solver = osqp.OSQP()
            solver.setup(P=P, q=q, verbose=self.verbosity > 2)
            res = solver.solve()

            new_b_estimate = res.x + b_estimate
            n_evals += res.info.iter

            if self.verbosity > 1:
                print('Estimated b:', new_b_estimate)

            if j > 0 and np.max(np.abs(new_b_estimate - b_estimate)) < self.tol:
                if self.verbosity > 1:
                    print('MM converged, stopping...')
                break
            b_estimate = new_b_estimate

        return b_estimate, n_evals, 0
