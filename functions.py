import random
import numpy as np
import scipy.optimize

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(X, y, s, test_size = 0.2):
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size = test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, s_train, s_test

def create_s(y, c): # c - label_frequency
    s = np.array(y)
    positives = np.where(s == 1)[0]
    
    unlabelled_samples = positives[np.random.random(len(positives)) < 1 - c]
    s[unlabelled_samples] = 0
    return s

def sigma(s):
    res = 1 / (1 + np.exp(-s))
    return res

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

def oracle_risk(b, X, y):
    X = np.insert(X, 0, 1, axis = 1)
    n = X.shape[0]

    probability = sigma(np.matmul(X, b))
    log_likelihood = np.sum(y * np.log(probability) + (1 - y) * np.log(1 - probability))

    return -log_likelihood / n

def oracle_risk_derivative(b, X, y):
    X = np.insert(X, 0, 1, axis = 1)
    n = X.shape[0]

    v = ((y - 1) * np.exp(np.matmul(X, b)) + y) / (1 + np.exp(np.matmul(X, b)))
    partial_res = np.sum(X * v[:, np.newaxis], axis = 0)

    return -partial_res / n

def oracle_method(X, y):
    b_init = np.random.random(X.shape[1] + 1) / 100

    res = scipy.optimize.minimize(
        fun = oracle_risk, 
        x0 = b_init, 
        method = 'BFGS', 
        args = (X, y), 
        jac = oracle_risk_derivative
    )

    return res.x





def joint_risk(params, X, s):
    X = np.insert(X, 0, 1, axis = 1)
    n = X.shape[0]

    b = params[:-1]
    c = params[-1]

    probability = c * sigma(np.matmul(X, b))
    log_likelihood = np.sum(s * np.log(probability) + (1 - s) * np.log(1 - probability))

    return -log_likelihood / n

def joint_risk_derivative(params, X, s):
    X = np.insert(X, 0, 1, axis = 1)
    n = X.shape[0]

    b = params[:-1]
    c = params[-1]

    ebx = np.exp(np.matmul(X, b))

    multiplier = - ((s - c) * ebx + s) / ( (1 + ebx) * ((c - 1) * ebx - 1) )
    partial_res = np.sum(X * multiplier[:, np.newaxis], axis = 0)

    derivative_wrt_c = np.sum((c * ebx - s * ebx - s) / (c * (c * ebx - ebx - 1)))
    partial_res = np.append(partial_res, derivative_wrt_c)

    return -partial_res / n

def joint_method(X, s):
    b_init = np.random.random(X.shape[1] + 2) / 100

    res = scipy.optimize.minimize(
        fun = joint_risk, 
        x0 = b_init, 
        method = 'L-BFGS-B', 
        args = (X, s), 
        jac = joint_risk_derivative
    )

    return res.x





def cccp_risk_wrt_b(b, X, s, c, b_prev):
    X_orig = X
    X = np.insert(X, 0, 1, axis = 1)
    n = X.shape[0]

    result = 0
    E_vex_part = oracle_risk(b, X_orig, s)
    result += E_vex_part

    # convex_part = result

    ebx = np.exp(np.matmul(X, b_prev))
    for j in range(len(b)):
        v = (1 - s) * (1 - c) * X[:, j] * ebx / (1 + (1 - c) * ebx)
        partial_res = b[j] * np.sum(v)
        result += -partial_res / n
    
    # print('RES:', result, 'CONVEX PART:', -convex_part / n, 'CONCAVE PART:', result + convex_part / n)
    return result

def cccp_risk_wrt_c(c, X, s, b):
    X = np.insert(X, 0, 1, axis = 1)
    n = X.shape[0]

    probability = c * sigma(np.matmul(X, b))
    log_likelihood = np.sum(s * np.log(probability) + (1 - s) * np.log(1 - probability))

    return -log_likelihood / n

def cccp_risk_derivative(X, s):
    pass

def cccp_method(X, s):
    b_init = np.random.random(X.shape[1] + 1) / 100
    c_init = 0.5

    b_estimate = b_init
    c_estimate = c_init
    print('b_init:', b_init)
    print('c_init:', c_init)

    for i in range(10):
        res = scipy.optimize.minimize(
            fun = cccp_risk_wrt_c, 
            x0 = c_estimate, 
            args = (X, s, b_estimate), 
            method = 'CG'
        )

        print('i:', i)
        print('c_estimate:', res.x)

        if c_estimate == res.x:
            break
        c_estimate = res.x

        for j in range(10):
            res = scipy.optimize.minimize(
                fun = cccp_risk_wrt_b, 
                x0 = b_estimate, 
                args = (X, s, c_estimate, b_estimate), 
                method = 'CG',
                options = {
                    'max_iter': 100
                }
            )

            print('j:', j)
            print('b_estimate:', res.x)

            if np.min(b_estimate == res.x) == 1:
                break
            b_estimate = res.x

    return np.append(b_estimate, c_estimate)
