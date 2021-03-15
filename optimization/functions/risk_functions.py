import numpy as np

from optimization.functions.helper_functions import sigma, add_bias


def oracle_risk(b, X, y):
    X = add_bias(X)
    n = X.shape[0]

    probability = sigma(np.matmul(X, b))
    # log_likelihood = np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability))

    # xb = np.matmul(X, b)
    # log_likelihood = np.sum(y * (xb - np.log(1 + np.exp(xb))) + (1 - y) * -np.log(1 + np.exp(xb)))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        non_summed_log_likelihood = \
            np.where(
                (y == 0) & (probability == 0),
                0,
                y * np.log(probability)
            ) + np.where(
                (y == 1) & (probability == 1),
                0,
                (1 - y) * np.log(1 - probability)
            )

        if np.sum(np.isnan(non_summed_log_likelihood)) > 0:
            warnings.warn("NaNs in result", RuntimeWarning)

        log_likelihood = np.sum(non_summed_log_likelihood)

    return -log_likelihood / n


def oracle_risk_derivative(b, X, y):
    X = add_bias(X)
    n = X.shape[0]

    v = ((y - 1) * np.exp(np.matmul(X, b)) + y) / (1 + np.exp(np.matmul(X, b)))
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

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        non_summed_log_likelihood = \
            np.where(
                (s == 0) & (probability == 0),
                0,
                s * np.log(probability)
            ) + np.where(
                (s == 1) & (probability == 1),
                0,
                (1 - s) * np.log(1 - probability)
            )

        if np.sum(np.isnan(non_summed_log_likelihood)) > 0:
            warnings.warn("NaNs in result", RuntimeWarning)

        log_likelihood = np.sum(non_summed_log_likelihood)

    return -log_likelihood / n


def joint_risk_derivative(params, X, s, exact_c=None):
    X = add_bias(X)
    n = X.shape[0]

    if exact_c is None:
        b = params[:-1]
        c = params[-1]
    else:
        b = params
        c = exact_c

    exb = np.exp(np.matmul(X, b))

    multiplier = - ((s - c) * exb + s) / ((1 + exb) * ((c - 1) * exb - 1))
    partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)

    if exact_c is None:
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
    # log_likelihood = np.sum(s * np.log(probability) + (1 - s) * np.log(1 - probability))

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")


        non_summed_log_likelihood = \
            np.where(
                (s == 0) & (probability == 0),
                0,
                s * np.log(probability)
            ) + np.where(
                (s == 1) & (probability == 1),
                0,
                (1 - s) * np.log(1 - probability)
            )

        if np.sum(np.isnan(non_summed_log_likelihood)) > 0:
            warnings.warn("NaNs in result", RuntimeWarning)

        log_likelihood = np.sum(non_summed_log_likelihood)

    return -log_likelihood / n


def cccp_risk_derivative_wrt_c(c, X, s, b):
    X = add_bias(X)
    n = X.shape[0]

    exb = np.exp(np.matmul(X, b))
    derivative_wrt_c = np.sum((c * exb - s * exb - s) / (c * (c * exb - exb - 1)))

    return -derivative_wrt_c / n
