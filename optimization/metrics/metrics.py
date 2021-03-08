import numpy as np


def approximation_error(oracle_pred, pred):
    return np.mean(np.abs(pred - oracle_pred))
