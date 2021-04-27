from optimization.c_estimation.base_c_estimator import BaseCEstimator


class PureAlphaEstimator(BaseCEstimator):
    def __init__(self, c_estimate: float, P_s_1: float):
        self.c_estimate = c_estimate
        self.P_s_1 = P_s_1

    def fit(self, X, s):
        pass  # intentional
