import dccp
import cvxpy as cvx
import numpy as np

from optimization.functions import add_bias
from optimization.__base_pu_classifier import BasePUClassifier


class DccpClassifier(BasePUClassifier):
    c_estimate: float

    def __init__(self, c: float):
        self.c = c

    def __build_problem(self, X, s):
        X = add_bias(X)
        n = X.shape[0]
        n_params = X.shape[1]

        b = cvx.Variable(n_params)
        t = cvx.Variable(1)

        expression = 0
        t_expression = 0

        expression += s * cvx.log(self.c)
        # e1 = s * cvx.log(self.c)

        expression += cvx.multiply(s, X @ b - cvx.logistic(X @ b))
        # e2 = cvx.multiply(s, X @ b - cvx.logistic(X @ b))

        expression += cvx.multiply(s - 1, cvx.logistic(X @ b))
        # e3 = cvx.multiply(s - 1, cvx.logistic(X @ b))

        t_expression += cvx.multiply(s - 1, cvx.logistic(cvx.log(1 - self.c) + X @ b))
        # e4 = cvx.multiply(1 - s, cvx.logistic(cvx.log(1 - self.c) + X @ b))

        # print(e1.curvature, e2.curvature, e3.curvature, e4.curvature)

        expression = cvx.sum(expression)
        expression *= -1 / n

        t_expression = cvx.sum(t_expression)
        t_expression *= 1 / n
        expression += -t

        problem = cvx.Problem(cvx.Minimize(expression), [t == t_expression])

        b.value = np.array(np.random.random(n_params) / 100)
        t.value = np.array([0])

        return problem, b

    def fit(self, X, s):
        problem, b = self.__build_problem(X, s)

        # print(problem.is_dcp(), dccp.is_dccp(problem))
        result = problem.solve(method='dccp', tau=0.05, verbose=True, max_iter=5,
                               max_iters=1000, ccp_times=1, solver=cvx.ECOS)

        print("problem status =", problem.status)
        print("b =", b.value)
        print("cost value =", result)

        self.params = b.value
        self.c_estimate = self.c

