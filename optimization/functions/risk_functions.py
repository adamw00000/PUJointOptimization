import sys
import warnings
import numpy as np
from sklearn.metrics import log_loss

from optimization.functions.helper_functions import sigma, add_bias


def oracle_risk(b, X, y):
    X = add_bias(X)
    n = X.shape[0]

    probability = sigma(np.matmul(X, b))
    # log_likelihood = np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability))

    # xb = np.matmul(X, b)
    # log_likelihood = np.sum(y * (xb - np.log(1 + np.exp(xb))) + (1 - y) * -np.log(1 + np.exp(xb)))

    return log_loss(y, probability, normalize=True)

def oracle_risk_derivative(b, X, y):
    X = add_bias(X)
    n = X.shape[0]

    def safe_v():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exb = np.exp(np.matmul(X, b))

            v = np.where(
                (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                y - 1,  # exb -> inf
                ((y - 1) * np.exp(np.matmul(X, b)) + y) / (1 + np.exp(np.matmul(X, b)))
            )

        if np.sum(np.isnan(v)) > 0 or np.sum(np.isinf(v)) > 0:
            warnings.warn("NaNs/inf in result", RuntimeWarning)

        return v

    # v = ((y - 1) * np.exp(np.matmul(X, b)) + y) / (1 + np.exp(np.matmul(X, b)))
    v = safe_v()
    partial_res = np.sum(X * v.reshape(-1, 1), axis=0)

    return -partial_res / n


def joint_risk(params, X, s, exact_c=None):
    X = add_bias(X)
    n = X.shape[0]

    if exact_c is None:
        b = params[:-1]
        c = params[-1]
    else:
        b = params
        c = exact_c

    probability = c * sigma(np.matmul(X, b))
    # log_likelihood = np.sum(s * np.log(probability) + (1 - s) * np.log(1 - probability))

    return log_loss(s, probability, normalize=True)


def joint_risk_derivative(params, X, s, exact_c=None):
    X = add_bias(X)
    n = X.shape[0]

    if exact_c is None:
        b = params[:-1]
        c = params[-1]
    else:
        b = params
        c = exact_c

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exb = np.exp(np.matmul(X, b))

    # multiplier = - ((s - c) * exb + s) / ((1 + exb) * ((c - 1) * exb - 1))
    # partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)

    def safe_multiplier():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            multiplier = np.where(
                (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                0,  # exb -> inf
                - ((s - c) * exb + s) / ((1 + exb) * ((c - 1) * exb - 1))
            )

        if np.sum(np.isnan(multiplier)) > 0 or np.sum(np.isinf(multiplier)) > 0:
            warnings.warn("NaNs/inf in result", RuntimeWarning)

        res = np.sum(X * multiplier.reshape(-1, 1), axis=0)
        return res

    partial_res = safe_multiplier()

    if exact_c is None:
        # derivative_wrt_c = np.sum((c * exb - s * exb - s) / (c * (c * exb - exb - 1)))

        def safe_derivative_wrt_c():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                partial = np.where(
                    exb == 0,
                    s / c,  # exb -> 0
                    np.where(
                        (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                        (c - s) / c * (c - 1),  # exb -> inf
                        (c * exb - s * exb - s) / (c * (c * exb - exb - 1))
                    )
                )

            if np.sum(np.isnan(partial)) > 0 or np.sum(np.isinf(partial)) > 0:
                warnings.warn("NaNs/inf in result", RuntimeWarning)

            return np.sum(partial)

        derivative_wrt_c = safe_derivative_wrt_c()
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
    n = X.shape[0]

    probability = c * sigma(np.matmul(X, b))
    # log_likelihood = np.sum(s * np.log(probability) + (1 - s) * np.log(1 - probability))

    return log_loss(s, probability, normalize=True)


def cccp_risk_derivative_wrt_c(c, X, s, b):
    X = add_bias(X)
    n = X.shape[0]

    exb = np.exp(np.matmul(X, b))
    # derivative_wrt_c = np.sum((c * exb - s * exb - s) / (c * (c * exb - exb - 1)))

    def safe_derivative_wrt_c():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            partial = np.where(
                exb == 0,
                s / c,  # exb -> 0
                np.where(
                    (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                    (c - s) / (c * (c - 1)),  # exb -> inf
                    (c * exb - s * exb - s) / (c * (c * exb - exb - 1))
                )
            )

        if np.sum(np.isnan(partial)) > 0 or np.sum(np.isinf(partial)) > 0:
            warnings.warn("NaNs/inf in result", RuntimeWarning)

        return np.sum(partial)

    derivative_wrt_c = safe_derivative_wrt_c()
    return -derivative_wrt_c / n
