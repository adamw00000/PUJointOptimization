# %%
import datasets
from data_preprocessing import create_case_control_dataset

target_c = 0.5
X, y = datasets.get_datasets()['spambase']

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

# %%
from optimization.em_cc_classifier import EmCcClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error

clf = EmCcClassifier()
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