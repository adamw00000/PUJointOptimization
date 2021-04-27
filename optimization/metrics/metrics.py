import numpy as np
from sklearn.metrics import roc_auc_score


def approximation_error(pred, oracle_pred):
    return np.mean(np.abs(pred - oracle_pred))


def c_error(c_estimate, c):
    return np.abs(c - c_estimate)


def auc(y, y_pred):
    return roc_auc_score(y, y_pred)


def alpha_error(alpha_estimate, y):
    alpha = np.mean(y == 1)
    return np.abs(alpha - alpha_estimate)
