# %%
import datasets
from data_preprocessing import create_case_control_dataset
from optimization.c_estimation import TIcEEstimator

target_c = 0.1
X, y = datasets.get_datasets()['credit-a']

X_new, y_new, s, c = create_case_control_dataset(X, y, target_c)

from data_preprocessing import preprocess

X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X_new, y_new, s, test_size=0.2)

from optimization import OracleClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import auc

clf = OracleClassifier()
clf.fit(X_train, y_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))
print('AUC:', auc(y_test, y_pred))

y_proba_oracle = y_proba

# %%
import numpy as np

n_p = np.sum(s == 1)
n_u = np.sum(s == 0)

alpha = np.mean(y_new == 0)
# a = n_p / (n_p + alpha * n_u)
a = (n_p + alpha * n_u) / (alpha * n_u)

print(a, c)

# %%
from optimization import JointClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error

clf = JointClassifier()
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
est_c = clf.c_estimate
print('Estimated c:', est_c)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('c error:', c_error(clf.c_estimate, c))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

import numpy as np
P_s_1 = np.mean(s == 1)
alpha = np.mean(y == 1)
est_alpha = (P_s_1 * (1 - est_c)) / (est_c * (1 - P_s_1))
correct_est_alpha = (P_s_1 * (1 - c)) / (c * (1 - P_s_1))
print(f'Alpha: {alpha}, estimated alpha: {est_alpha}, error: {np.abs(est_alpha - alpha)}, '
      f'correct estimate error: {np.abs(correct_est_alpha - alpha)}')

print(clf.get_CC_alpha())

# %%
from optimization.em_cc_classifier import EmCcClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error

clf = EmCcClassifier(TIcEEstimator())
clf.fit(X_train, s_train, c)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
# est_c = clf.c_estimate
# print('Estimated c:', est_c)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

# print('c error:', c_error(clf.c_estimate, c))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))
