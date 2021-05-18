import numpy as np
from optimization.functions import joint_risk, joint_risk_derivative, cccp_risk_wrt_b, cccp_risk_derivative_wrt_b, \
    sigma, add_bias


def joint_risk_with_info(params, X, s, info, exact_c=None):
    res = joint_risk(params, X, s, exact_c)

    if exact_c is None:
        b = params[:-1]
        c = params[-1]
    else:
        b = params
        c = exact_c

    # X_bias = add_bias(X)
    # sig = sigma(np.matmul(X_bias, b))
    # print('Min proba:', np.min(sig), 'Max proba:', np.max(sig))
    # print('LL:', res * -5000)

    info['param_history'].append(b)
    info['risk_values'].append(res)
    info['c_history'].append(c)
    return res


def joint_risk_derivative_with_info(params, X, s, info, exact_c=None):
    return joint_risk_derivative(params, X, s, exact_c)


def cccp_risk_wrt_b_with_info(b, X, s, c, b_prev, info):
    res = cccp_risk_wrt_b(b, X, s, c, b_prev)
    info['param_history'].append(b)
    info['risk_values'].append(joint_risk(b, X, s, c))
    return res


def cccp_risk_derivative_wrt_b_with_info(b, X, s, c, b_prev, info):
    return cccp_risk_derivative_wrt_b(b, X, s, c, b_prev)
