import numpy as np

from optimization.functions.helper_functions import sigma, add_bias


def oracle_risk(b, X, y):
    X = add_bias(X)
    n = X.shape[0]

    probability = sigma(np.matmul(X, b))
    # ###
    #     import sys
    #     # np.set_printoptions(threshold=sys.maxsize)
    #     # print(probability)
    #     print(np.sum(probability == 0), np.sum(probability < 0), np.sum(probability == 1), np.sum(probability > 1))
    #     if np.sum(probability == 1):
    #         np.set_printoptions(threshold=sys.maxsize)
    #         print(np.max(np.matmul(X, b)))
    # ###
    #     import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")

    log_likelihood = np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    return -log_likelihood / n


def oracle_risk_derivative(b, X, y):
    X = add_bias(X)
    n = X.shape[0]

    v = ((y - 1) * np.exp(np.matmul(X, b)) + y) / (1 + np.exp(np.matmul(X, b)))
    partial_res = np.sum(X * v.reshape(-1, 1), axis=0)

    return -partial_res / n


def joint_risk(params, X, s):
    X = add_bias(X)
    n = X.shape[0]

    b = params[:-1]
    c = params[-1]

    probability = c * sigma(np.matmul(X, b))

    # import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    log_likelihood = np.sum(s * np.log(probability) + (1 - s) * np.log(1 - probability))
    return -log_likelihood / n


def joint_risk_derivative(params, X, s):
    X = add_bias(X)
    n = X.shape[0]

    b = params[:-1]
    c = params[-1]

    exb = np.exp(np.matmul(X, b))

    # import warnings
    # with warnings.catch_warnings():
    #     warnings.simplefilter("ignore")
    multiplier = - ((s - c) * exb + s) / ((1 + exb) * ((c - 1) * exb - 1))
    partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)

    derivative_wrt_c = np.sum((c * exb - s * exb - s) / (c * (c * exb - exb - 1)))
    partial_res = np.append(partial_res, derivative_wrt_c)

    return -partial_res / n


def cccp_risk_wrt_b(b, X, s, c, b_prev):
    X_orig = X
    X = add_bias(X)
    n = X.shape[0]

    result = 0
    E_vex_part = oracle_risk(b, X_orig, s)
    result += E_vex_part

    exb = np.exp(np.matmul(X, b_prev))
    for j in range(len(b)):
        v = (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
        partial_res = b[j] * np.sum(v)
        result += -partial_res / n

    return result


def cccp_risk_derivative_wrt_b(b, X, s, c, b_prev):
    X_orig = X
    X = add_bias(X)
    n = X.shape[0]

    E_vex_part = oracle_risk_derivative(b, X_orig, s)
    result = E_vex_part

    exb = np.exp(np.matmul(X, b_prev))
    for j in range(len(b)):
        v = (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
        partial_res = np.sum(v)
        result[j] += -partial_res / n

    return result


def cccp_risk_wrt_c(c, X, s, b):
    X = add_bias(X)
    n = X.shape[0]

    probability = c * sigma(np.matmul(X, b))
    log_likelihood = np.sum(s * np.log(probability) + (1 - s) * np.log(1 - probability))

    return -log_likelihood / n


def cccp_risk_derivative_wrt_c(c, X, s, b):
    X = add_bias(X)
    n = X.shape[0]

    exb = np.exp(np.matmul(X, b))
    derivative_wrt_c = np.sum((c * exb - s * exb - s) / (c * (c * exb - exb - 1)))

    return -derivative_wrt_c / n
