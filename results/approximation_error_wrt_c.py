import datasets
import numpy as np
import matplotlib.pyplot as plt

from data_preprocessing import create_s, preprocess
from optimization import CccpClassifier, JointClassifier, OracleClassifier
from optimization.metrics import approximation_error


def oracle_prediction(X_train, y_train, X_test):
    clf = OracleClassifier()
    clf.fit(X_train, y_train.to_numpy())

    y_proba = clf.predict_proba(X_test)
    return y_proba


def joint_prediction(X_train, s_train, X_test):
    clf = JointClassifier()
    clf.fit(X_train, s_train)

    y_proba = clf.predict_proba(X_test)
    return y_proba


def cccp_prediction(X_train, s_train, X_test):
    clf = CccpClassifier()
    clf.fit(X_train, s_train)

    y_proba = clf.predict_proba(X_test)
    return y_proba


if __name__ == '__main__':
    X, y = datasets.load_spambase()

    fig = plt.figure()
    ax = plt.gca()

    joint_errors = []
    cccp_errors = []

    c_values = np.arange(0.1, 1, 0.1)
    for c in c_values:
        joint_errors_wrt_c = []
        cccp_errors_wrt_c = []

        for i in range(10):
            s = create_s(y, c)
            X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0.2)

            oracle_pred = oracle_prediction(X_train, y_train, X_test)
            joint_pred = joint_prediction(X_train, s_train, X_test)
            cccp_pred = cccp_prediction(X_train, s_train, X_test)

            joint_error = approximation_error(oracle_pred, joint_pred)
            cccp_error = approximation_error(oracle_pred, cccp_pred)

            joint_errors_wrt_c.append(joint_error)
            cccp_errors_wrt_c.append(cccp_error)

        mean_joint_error = np.mean(joint_errors_wrt_c)
        mean_cccp_error = np.mean(cccp_errors_wrt_c)
        joint_errors.append(mean_joint_error)
        cccp_errors.append(mean_cccp_error)

    ax.plot(c_values, joint_errors)
    ax.plot(c_values, cccp_errors)

    ax.scatter(c_values, joint_errors)
    ax.scatter(c_values, cccp_errors)

    plt.legend(['Joint error', 'CCCP error'])
    plt.xlabel(r'Label frequency $c$')
    plt.ylabel('Approximation error (AE) for posterior')

    plt.savefig('res.png', dpi=150, bbox_inches='tight')
    plt.show()


