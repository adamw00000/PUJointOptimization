import time
import numpy as np

from optimization import OracleClassifier
from optimization.c_estimation.base_c_estimator import BaseCEstimator


class ElkanNotoEstimator(BaseCEstimator):
    time: float

    def fit(self, X, s):
        self.P_s_1 = float(np.mean(s == 1))

        ti = time.time()
        clf = OracleClassifier()
        clf.fit(X, s)

        positive_samples = np.where(s == 1)[0]
        X_pos = X[positive_samples, :]

        proba = clf.predict_proba(X_pos)
        ti = time.time() - ti

        self.c_estimate = float(np.mean(proba))
        self.time = ti
        return self.c_estimate
