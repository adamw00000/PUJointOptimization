import dccp
import mosek
import cvxpy as cvx
import numpy as np
import numpy.typing as npt

from optimization.functions import add_bias
from optimization.__split_optimization_pu_classifier import SplitOptimizationPUClassifier


class DccpClassifier(SplitOptimizationPUClassifier):
    tau: float

    def __init__(self, tol: float = 1e-10, max_iter: int = 100, dccp_max_iter: int = 1000, tau: float = 1,
                 verbosity: int = 0):
        super().__init__('DCCP', tol=tol, max_iter=max_iter, max_inner_iter=dccp_max_iter, verbosity=verbosity)
        self.tau = tau

    @staticmethod
    def __build_problem(X, s, c_estimate, old_b_estimate):
        X = add_bias(X)
        n = X.shape[0]
        n_params = X.shape[1]

        b = cvx.Variable(n_params)
        t = cvx.Variable(1)

        f1 = s * cvx.log(c_estimate)
        # f2_1 = cvx.multiply(s, X @ b - cvx.logistic(X @ b))
        # f2_2 = cvx.multiply(s - 1, cvx.logistic(X @ b))
        f2 = cvx.multiply(s, X @ b) - cvx.logistic(X @ b)
        f3 = cvx.multiply(s - 1, cvx.logistic(cvx.log(1 - c_estimate) + X @ b))

        # print(f1.curvature, f2_1.curvature, f2_2.curvature, f3.curvature)

        expression = -1/n * cvx.sum(f1 + f2) - t
        t_expression = -1/n * cvx.sum(f3)

        problem = cvx.Problem(cvx.Minimize(expression), [t == t_expression])

        b.value = old_b_estimate
        t.value = np.array([0])

        return problem, b

    def _minimize_wrt_b(self, X, s, c_estimate, old_b_estimate) -> npt.ArrayLike:
        problem, b = self.__build_problem(X, s, c_estimate, old_b_estimate)

        # print(problem.is_dcp(), dccp.is_dccp(problem))
        result = problem.solve(method='dccp', tau=self.tau, solver=cvx.MOSEK, max_iter=self.max_inner_iter,
                               verbose=True if self.verbosity > 1 else False)
        # result = problem.solve(method='dccp', tau=self.tau, verbose=True, max_iter=5,
        #                        max_iters=100, ccp_times=1, solver=cvx.ECOS)

        print("DCCP problem status:", problem.status)
        # print("b =", b.value)
        # print("cost value =", result[0])

        return b.value
