import sys
import warnings
import numpy as np

from optimization.functions.helper_functions import sigma, add_bias

# from sklearn.metrics import log_loss
def my_log_loss(y_true, y_pred, *, eps=1e-15, sample_weight=None):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    y_pred = np.clip(y_pred, eps, 1 - eps)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
    loss = -(y_true * np.log(y_pred)).sum(axis=1)

    return np.average(loss, weights=sample_weight)


def oracle_risk(b, X, y):
    X = add_bias(X)

    probability = sigma(np.matmul(X, b))
    return my_log_loss(y, probability)


def oracle_risk_derivative(b, X, y):
    X = add_bias(X)
    n = X.shape[0]

    sig = sigma(np.matmul(X, b))

    multiplier = y - sig
    partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)

    return -partial_res / n


def joint_risk(params, X, s, exact_c=None):
    X = add_bias(X)

    if exact_c is None:
        b = params[:-1]
        c = params[-1]
    else:
        b = params
        c = exact_c

    probability = c * sigma(np.matmul(X, b))
    return my_log_loss(s, probability)


def joint_risk_derivative(params, X, s, exact_c=None):
    X = add_bias(X)
    n = X.shape[0]

    if exact_c is None:
        b = params[:-1]
        c = params[-1]
    else:
        b = params
        c = exact_c

    sig = sigma(np.matmul(X, b))
    multiplier = (1 - sig) * (s - c * sig) / (c * (1 - c * sig))

    partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)

    if exact_c is None:
        derivative_wrt_c = np.sum(s / c + (s - 1) * sig / (1 - c * sig))
        partial_res = np.append(partial_res, derivative_wrt_c)

    return -partial_res / n


# def mm_q(b, X, s, c=None):
#     X = add_bias(X)
#     n = X.shape[0]
#
#     exb = np.exp(np.matmul(X, b))
#     multiplier = - ((s - c) * exb + s) / ((1 + exb) * ((c - 1) * exb - 1))
#     partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)
#
#     return -partial_res / n


def cccp_risk_wrt_b(b, X, s, c, b_prev):
    X_orig = X
    X = add_bias(X)
    n = X.shape[0]

    result = 0
    E_vex_part = oracle_risk(b, X_orig, s)
    result += E_vex_part
    # print('Half CCCP risk value:', result)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exb = np.exp(np.matmul(X, b_prev))

    def safe_v():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            v = np.where(
                exb == 0,
                s / c,  # exb -> 0
                np.where(
                    (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                    (1 - s) * (1 - c) * X[:, j] / (1 - c),  # exb -> inf
                    (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
                )
            )

        if np.sum(np.isnan(v)) > 0 or np.sum(np.isinf(v)) > 0:
            warnings.warn("NaNs/inf in result", RuntimeWarning)

        return v

    for j in range(len(b)):
        # v = (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
        v = safe_v()

        partial_res = b[j] * np.sum(v)
        result += -partial_res / n

    # print('CCCP risk value:', result)
    return result


def cccp_risk_derivative_wrt_b(b, X, s, c, b_prev):
    X_orig = X
    X = add_bias(X)
    n = X.shape[0]

    E_vex_part = oracle_risk_derivative(b, X_orig, s)
    result = E_vex_part

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exb = np.exp(np.matmul(X, b_prev))

    def safe_v():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            v = np.where(
                exb == 0,
                s / c,  # exb -> 0
                np.where(
                    (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                    (1 - s) * (1 - c) * X[:, j] / (1 - c),  # exb -> inf
                    (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
                )
            )

        if np.sum(np.isnan(v)) > 0 or np.sum(np.isinf(v)) > 0:
            warnings.warn("NaNs/inf in result", RuntimeWarning)

        return v

    for j in range(len(b)):
        # v = (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
        v = safe_v()
        partial_res = np.sum(v)
        result[j] += -partial_res / n

    return result


def cccp_risk_wrt_c(c, X, s, b):
    X = add_bias(X)
    probability = c * sigma(np.matmul(X, b))

    return my_log_loss(s, probability)


def cccp_risk_derivative_wrt_c(c, X, s, b):
    X = add_bias(X)
    n = X.shape[0]

    sig = sigma(np.matmul(X, b))
    derivative_wrt_c = np.sum(s / c + (s - 1) * sig / (1 - c * sig))
    return -derivative_wrt_c / n
