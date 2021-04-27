class BaseCEstimator:
    P_s_1: float  # P(s = 1)
    c_estimate: float

    def get_STD_alpha(self):
        alpha = self.P_s_1 / self.c_estimate
        return max(0.0, min(1.0, alpha))

    def get_CC_alpha(self):
        alpha = (self.P_s_1 * (1 - self.c_estimate)) / (self.c_estimate * (1 - self.P_s_1))
        return max(0.0, min(1.0, alpha))
