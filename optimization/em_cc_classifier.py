import time
import numpy as np
import scipy.optimize

from optimization.c_estimation import BaseCEstimator
from optimization.functions import oracle_risk, oracle_risk_derivative, sigma, add_bias
from optimization.__base_pu_classifier import BasePUClassifier


class EmCcClassifier(BasePUClassifier):
    adjustment: float
    c_estimator: BaseCEstimator
    max_iter: int
    tol: float
    verbosity: int
    include_bias: bool

    def __init__(self, c_estimator: BaseCEstimator, max_iter: int = 1000,
                 tol: float = 1e-4, verbosity: int = 1, include_bias: bool = True):
        self.c_estimator = c_estimator
        self.max_iter = max_iter
        self.tol = tol
        self.verbosity = verbosity
        self.include_bias = include_bias

    def fit(self, X, s, c: float = None):
        t = time.time()
        if self.include_bias:
            X = add_bias(X)
        self.P_S_1 = np.mean(s == 1)
        self.evaluations = 0

        if c is None:
            self.c_estimate = self.c_estimator.fit(X, s)
        else:
            self.c_estimate = c

        alpha = self.P_S_1 / self.c_estimate

        y_estimate = np.where(s == 1, 1, alpha)
        # y_estimate = np.repeat(alpha, len(s))
        b_estimate = np.zeros(X.shape[1])

        n_p = np.sum(s == 1)
        n_u = np.sum(s == 0)

        self.adjustment = np.log((n_p + alpha * n_u) / (alpha * n_u))

        for i in range(self.max_iter):
            res = scipy.optimize.minimize(
                fun=oracle_risk,
                x0=b_estimate,
                method='BFGS',
                args=(X, y_estimate),
                jac=oracle_risk_derivative
            )

            new_b_estimate = res.x
            self.evaluations += res.nfev + res.njev

            eta = np.matmul(X_with_bias, new_b_estimate)
            eta = eta - self.adjustment

            if i > 0 and np.min(np.abs(new_b_estimate - b_estimate)) < self.tol:
                if self.verbosity > 1:
                    print('EM converged, stopping...')
                self.iterations = i + 1
                break

            b_estimate = new_b_estimate
            y_estimate = np.where(s == 1, 1, sigma(eta))

        self.params = b_estimate
        self.total_time = time.time() - t

    def predict_proba(self, X):
        X = add_bias(X)
        eta = np.matmul(X, self.params)
        eta = eta - self.adjustment

        y_proba = sigma(eta)
        return y_proba
