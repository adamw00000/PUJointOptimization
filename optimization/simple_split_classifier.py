import scipy.optimize
import numpy as np
from optimization.__split_optimization_pu_classifier import SplitOptimizationPUClassifier
from optimization.functions import joint_risk, joint_risk_derivative


class SimpleSplitClassifier(SplitOptimizationPUClassifier):
    inner_tol: float

    def __init__(self, tol: float = 1e-4, max_iter: int = 100, b_opt_max_iter: int = 1000, inner_max_iter: int = 1000,
                 verbosity: int = 0, get_info: bool = False, reset_params_each_iter: bool = True,
                 include_bias: bool = True, inner_tol: float = 1e-10):
        super().__init__('Simple split', tol=tol, max_iter=max_iter, max_inner_iter=b_opt_max_iter, verbosity=verbosity,
                         get_info=get_info, reset_params_each_iter=reset_params_each_iter, include_bias=include_bias)
        self.inner_tol = inner_tol

    def _minimize_wrt_b(self, X, s, c_estimate, old_b_estimate):
        n_fevals = 0
        n_jevals = 0

        risk_values = []
        param_history = []

        if self.reset_params_each_iter:
            b_estimate = np.zeros_like(old_b_estimate)
        else:
            b_estimate = old_b_estimate

        for j in range(self.max_inner_iter):
            convergence_info = {'param_history': [], 'risk_values': []}

            if self.verbosity > 1:
                print(f'{self.inner_method_name} step:', f'{j + 1}/{self.max_inner_iter}')

            res = scipy.optimize.minimize(
                fun=joint_risk,
                jac=joint_risk_derivative,
                x0=b_estimate,
                args=(X, s, c_estimate),
                method='CG',
                # method='BFGS',
                options={
                    # 'maxiter': self.inner_max_iter,
                    # 'disp': self.verbosity > 1,
                }
            )
            new_b_estimate = res.x

            if self.get_info:
                param_history += convergence_info['param_history']
                risk_values += convergence_info['risk_values']

            n_fevals += res.nfev
            n_jevals += getattr(res, 'njev', 0)

            if self.verbosity > 1:
                print('Estimation success:', res.success, f'({res.nit} iterations)')
                if not res.success:
                    print('Solver status:', res.status)
                    print('Message:', res.message)
                print('Estimated b:', new_b_estimate)

            if j > 0 and np.sum(np.abs(new_b_estimate - b_estimate)) < self.inner_tol:
                if self.verbosity > 1:
                    print(f'{self.inner_method_name} converged, stopping...')
                break
            b_estimate = new_b_estimate

        return b_estimate, n_fevals, n_jevals, {
            'risk_values': risk_values,
            'param_history': param_history,
            'c_history': list(np.repeat(c_estimate, len(risk_values))),
        }
