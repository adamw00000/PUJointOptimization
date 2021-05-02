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


def my_log_loss_for_weighted(y_true, y_pred, *, eps=1e-15, w_pos_pos, w_pos_neg, w_neg_pos, w_neg_neg):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)

    weight_matrix_neg = np.where(y_true == 1, w_pos_neg, w_neg_neg)
    weight_matrix_pos = np.where(y_true == 1, w_pos_pos, w_neg_pos)
    weight_matrix = np.append(weight_matrix_neg, weight_matrix_pos, axis=1)

    y_pred = np.clip(y_pred, eps, 1 - eps)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    if y_pred.shape[1] == 1:
        y_pred = np.append(1 - y_pred, y_pred, axis=1)

    y_pred /= y_pred.sum(axis=1)[:, np.newaxis]

    loss = -(weight_matrix * np.log(y_pred)).sum(axis=1)

    return np.average(loss)


def weighted_risk(b, X, s, c):
    X = add_bias(X)
    probability = sigma(np.matmul(X, b))

    # labeled_examples = np.where(s == 1)[0]
    # neg_weight_labeled_samples = s[labeled_examples]
    # s_full = np.append(s, neg_weight_labeled_samples)
    #
    # base_weights = np.where(s == 0, 1, 1 / c)
    # neg_weights = (1 - 1 / c) * np.ones(len(neg_weight_labeled_samples))
    # weights = np.append(base_weights, neg_weights)
    #
    # neg_weight_labeled_probas = probability[labeled_examples]
    # probability_full = np.append(probability, neg_weight_labeled_probas)
    #
    # return my_log_loss(s_full, probability_full, sample_weight=weights)

    return my_log_loss_for_weighted(s, probability,
                                    w_pos_pos=1/c,
                                    w_pos_neg=1 - 1/c,
                                    w_neg_neg=1,
                                    w_neg_pos=0)


def weighted_risk_derivative(b, X, s, c):
    X = add_bias(X)
    n = X.shape[0]

    sig = sigma(np.matmul(X, b))

    multiplier = s / c - sig
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

    # ### ORIGINAL R IMPLEMENTATION START

    # term1 = c * sigma(np.matmul(X, b))
    # term2 = 1 - c * sigma(np.matmul(X, b))
    #
    # term1 = np.where(term1 < 0, 0, term1)
    # term2 = np.where(term2 < 0, 0, term2)
    #
    # n = X.shape[0]
    # res = -1/n * np.sum(s * np.log(term1) + (1 - s) * np.log(term2))
    # return res

    # ### ORIGINAL R IMPLEMENTATION END

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

    # ### ORIGINAL R IMPLEMENTATION START

    # var1 = sig * (1 - sig)
    # sigma1 = sig
    #
    # a = var1 * ((s - c * sigma1) / (sigma1 * (1 - c * sigma1)))
    #
    # res = np.sum(X * a.reshape(-1, 1), axis=0)
    #
    # if exact_c is None:
    #     gr_c1 = sum(-s / c + (s - 1) * sigma1 / (1 - c * sigma1))
    #     res = np.append(res, gr_c1)
    #
    # return -res/n

    # ### ORIGINAL R IMPLEMENTATION END

    multiplier = (1 - sig) * (s - c * sig) / (c * (1 - c * sig))

    partial_res = np.sum(X * multiplier.reshape(-1, 1), axis=0)

    if exact_c is None:
        derivative_wrt_c = np.sum(s / c + (s - 1) * sig / (1 - c * sig))
        partial_res = np.append(partial_res, derivative_wrt_c)

    return -partial_res / n


def mm_q(b, X, s, c):
    n = X.shape[0]
    return n * joint_risk_derivative(b, X, s, exact_c=c)


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
                (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                (1 - s) * X[:, j],  # exb -> inf
                (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
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
                (exb > np.sqrt(sys.float_info.max)) | (np.isinf(exb)),
                (1 - s) * X[:, j],  # exb -> inf
                (1 - s) * (1 - c) * X[:, j] * exb / (1 + (1 - c) * exb)
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
