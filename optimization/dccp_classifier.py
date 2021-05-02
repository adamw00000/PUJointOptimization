import dccp
import mosek
import cvxpy as cvx
import numpy as np
import numpy.typing as npt

from optimization.functions import add_bias, joint_risk
from optimization.__split_optimization_pu_classifier import SplitOptimizationPUClassifier


class DccpClassifier(SplitOptimizationPUClassifier):
    tau: float
    mosek_max_iter: int

    def __init__(self, tol: float = 1e-4, max_iter: int = 100, dccp_max_iter: int = 1000, tau: float = 1,
                 mosek_max_iter: int = 1000, verbosity: int = 0, get_info: bool = False,
                 reset_params_each_iter: bool = False):
        super().__init__('DCCP', tol=tol, max_iter=max_iter, max_inner_iter=dccp_max_iter, verbosity=verbosity,
                         get_info=get_info, reset_params_each_iter=reset_params_each_iter)
        self.tau = tau
        self.mosek_max_iter = mosek_max_iter

    def __build_problem(self, X, s, c_estimate, old_b_estimate):
        X = add_bias(X)
        n = X.shape[0]
        n_params = X.shape[1]

        b = cvx.Variable(n_params)
        t = cvx.Variable(1)

        f1 = s * cvx.log(c_estimate)
        f2 = cvx.multiply(s, X @ b) - cvx.logistic(X @ b)
        f3 = cvx.multiply(s - 1, cvx.logistic(cvx.log(1 - c_estimate) + X @ b))
        # print(f1.curvature, f2.curvature, f3.curvature)

        expression = -1/n * cvx.sum(f1 + f2) - t
        t_expression = -1/n * cvx.sum(f3)

        problem = cvx.Problem(cvx.Minimize(expression), [t == t_expression])

        if self.reset_params_each_iter:
            b.value = np.zeros_like(old_b_estimate)
        else:
            b.value = old_b_estimate
        t.value = np.array([0])

        return problem, b

    def _minimize_wrt_b(self, X, s, c_estimate, old_b_estimate) -> (npt.ArrayLike, int, int):
        problem, b = self.__build_problem(X, s, c_estimate, old_b_estimate)

        # print(problem.is_dcp(), dccp.is_dccp(problem))
        result = problem.solve(method='dccp', tau=self.tau, solver=cvx.MOSEK,
                               max_iter=self.max_inner_iter,
                               verbose=True if self.verbosity > 1 else False,
                               mosek_params={
                                   'MSK_IPAR_INTPNT_MAX_ITERATIONS': self.mosek_max_iter
                               })

        if self.verbosity > 0:
            print("DCCP problem status:", problem.status)

        # n_evals = problem.solver_stats().num_iters
        n_evals = self.max_inner_iter

        return b.value, n_evals, 0, {
            'risk_values': [joint_risk(b.value, X, s, c_estimate)] if self.get_info else [],
            'param_history': [b.value] if self.get_info else [],
        }
