import numpy as np
import numpy.typing as npt
import osqp
from scipy.sparse import csc_matrix

from optimization.__split_optimization_pu_classifier import SplitOptimizationPUClassifier
from optimization.functions import mm_q, add_bias, joint_risk


class MMClassifier(SplitOptimizationPUClassifier):
    osqp_max_iter: int

    def __init__(self, tol: float = 1e-4, max_iter: int = 100, mm_max_iter: int = 1000,
                 osqp_max_iter: int = 4000, verbosity: int = 0, reset_params_each_iter: bool = True,
                 get_info: bool = False):
        super().__init__('MM', tol=tol, max_iter=max_iter, max_inner_iter=mm_max_iter, verbosity=verbosity,
                         get_info=get_info, reset_params_each_iter=reset_params_each_iter)
        self.osqp_max_iter = osqp_max_iter

    def _minimize_wrt_b(self, X, s, c, old_b_estimate) -> (npt.ArrayLike, int, int):
        n_evals = 0
        param_history = []
        risk_values = []
        c_history = []

        if self.reset_params_each_iter:
            b_estimate = np.zeros_like(old_b_estimate)
        else:
            b_estimate = old_b_estimate

        param_history.append(b_estimate)
        risk_values.append(joint_risk(b_estimate, X, s, c))

        X_with_bias = add_bias(X)
        P = csc_matrix(np.matmul(X_with_bias.T, X_with_bias) / 4)

        for j in range(self.max_inner_iter):
            q = mm_q(b_estimate, X, s, c)

            solver = osqp.OSQP()
            solver.setup(P=P, q=q, verbose=self.verbosity > 2,
                         max_iter=self.osqp_max_iter)
            res = solver.solve()

            new_b_estimate = res.x + b_estimate
            n_evals += res.info.iter

            param_history.append(new_b_estimate)
            risk_values.append(joint_risk(new_b_estimate, X, s, c))
            c_history.append(c)

            if self.verbosity > 1:
                print('Estimated b:', new_b_estimate)

            if j > 0 and np.max(np.abs(new_b_estimate - b_estimate)) < self.tol:
                if self.verbosity > 1:
                    print('MM converged, stopping...')
                break
            b_estimate = new_b_estimate

        return b_estimate, n_evals, 0, {
            'risk_values': risk_values,
            'param_history': param_history,
            'c_history': c_history,
        }
