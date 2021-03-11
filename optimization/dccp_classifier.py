import dccp
import cvxpy as cvx
import numpy as np

from optimization.functions import add_bias
from optimization.__base_pu_classifier import BasePUClassifier


class DccpClassifier(BasePUClassifier):
    c_estimate: float
    tau: float

    def __init__(self, c: float, tau: float = 1):
        self.c = c   # temporary
        self.tau = tau

    def __build_problem(self, X, s):
        X = add_bias(X)
        n = X.shape[0]
        n_params = X.shape[1]

        b = cvx.Variable(n_params)
        t = cvx.Variable(1)

        f1 = s * cvx.log(self.c)
        # f2_1 = cvx.multiply(s, X @ b - cvx.logistic(X @ b))
        # f2_2 = cvx.multiply(s - 1, cvx.logistic(X @ b))
        f2 = cvx.multiply(s, X @ b) - cvx.logistic(X @ b)
        f3 = cvx.multiply(s - 1, cvx.logistic(cvx.log(1 - self.c) + X @ b))

        # print(f1.curvature, f2_1.curvature, f2_2.curvature, f3.curvature)

        # expression = -cvx.sum(f1 + f2_1 + f2_2) / n
        expression = -1/n * cvx.sum(f1 + f2) - t
        t_expression = -1/n * cvx.sum(f3)

        problem = cvx.Problem(cvx.Minimize(expression), [t == t_expression])

        b.value = np.array(np.random.random(n_params) / 100)
        t.value = np.array([0])

        return problem, b

    def fit(self, X, s):
        problem, b = self.__build_problem(X, s)

        # print(problem.is_dcp(), dccp.is_dccp(problem))
        result = problem.solve(method='dccp', tau=self.tau, verbose=True, solver=cvx.ECOS)
        # result = problem.solve(method='dccp', tau=self.tau, verbose=True, max_iter=5,
        #                        max_iters=100, ccp_times=1, solver=cvx.ECOS)

        print("problem status =", problem.status)
        print("b =", b.value)
        print("cost value =", result[0])

        self.params = b.value
        self.c_estimate = self.c
