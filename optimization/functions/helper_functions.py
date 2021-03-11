import numpy as np


def sigma(s):
    return 1 / (1 + np.exp(-s))


def predict_proba(X, parameters):
    b0 = parameters[0]
    b = parameters[1:]
    proba = sigma(np.matmul(X, b) + b0)
    return proba


def predict(X, parameters):
    proba = predict_proba(X, parameters)
    y_pred = np.where(proba > 0.5, 1, 0)
    return y_pred


def accuracy(y_pred, y_test):
    return np.mean(y_pred == y_test)


def add_bias(X: np.array) -> np.array:
    return np.insert(X, 0, 1, axis=1)
