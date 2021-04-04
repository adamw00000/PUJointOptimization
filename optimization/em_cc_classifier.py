import numpy as np
import scipy.optimize

from optimization.functions import oracle_risk, oracle_risk_derivative, sigma, add_bias
from optimization.__base_pu_classifier import BasePUClassifier


class EmCcClassifier(BasePUClassifier):
    adjustment: float

    def maximization(self):
        pass

    def fit(self, X, s, c: float):
        P_s_eq_1 = np.mean(s == 1)
        alpha = P_s_eq_1 / c

        y_estimate = np.where(s == 1, 1, alpha)
        # y_estimate = np.repeat(alpha, len(s))
        b_estimate = np.zeros(X.shape[1] + 1)

        n_p = np.sum(s == 1)
        n_u = np.sum(s == 0)

        self.adjustment = np.log((n_p + alpha * n_u) / (alpha * n_u))
        X_with_bias = add_bias(X)

        for i in range(1000):
            res = scipy.optimize.minimize(
                fun=oracle_risk,
                x0=b_estimate,
                method='BFGS',
                args=(X, y_estimate),
                jac=oracle_risk_derivative
            )

            new_b_estimate = res.x
            eta = np.matmul(X_with_bias, new_b_estimate)
            eta = eta - self.adjustment

            b_estimate = new_b_estimate
            y_estimate = np.where(s == 1, 1, sigma(eta))

        self.params = b_estimate

    def predict_proba(self, X):
        X = add_bias(X)
        eta = np.matmul(X, self.params)
        eta = eta - self.adjustment

        y_proba = sigma(eta)
        return y_proba
