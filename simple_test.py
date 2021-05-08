# %%
import datasets
from data_preprocessing import create_s

target_c = 0.5
X, y = datasets.get_datasets()['spambase']
s, c = create_s(y, target_c)

from data_preprocessing import preprocess

X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0.2)

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
from optimization import NaiveClassifier
from optimization.c_estimation import TIcEEstimator
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import auc, approximation_error, c_error, alpha_error

clf = NaiveClassifier(TIcEEstimator())
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

print(f'Stats - time: {clf.total_time}, iterations: {clf.iterations}, evaluations: {clf.evaluations}')
b = clf.get_params()
est_c = clf.c_estimate
print('Estimated c:', est_c)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('')
print('c error:', c_error(clf.c_estimate, c))
print('alpha error:', alpha_error(clf.get_STD_alpha(), y))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization import WeightedClassifier
from optimization.c_estimation import TIcEEstimator
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import auc, approximation_error, c_error, alpha_error

clf = WeightedClassifier(TIcEEstimator())
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

print(f'Stats - time: {clf.total_time}, iterations: {clf.iterations}, evaluations: {clf.evaluations}')
b = clf.get_params()
est_c = clf.c_estimate
print('Estimated c:', est_c)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('')
print('c error:', c_error(clf.c_estimate, c))
print('alpha error:', alpha_error(clf.get_STD_alpha(), y))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization import JointClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error, alpha_error

clf = JointClassifier()
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

print(f'Stats - time: {clf.total_time}, iterations: {clf.iterations}, evaluations: {clf.evaluations}')
b = clf.get_params()
est_c = clf.c_estimate
print('Estimated c:', est_c)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('')
print('c error:', c_error(clf.c_estimate, c))
print('alpha error:', alpha_error(clf.get_STD_alpha(), y))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization import CccpClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error, alpha_error

clf = CccpClassifier(verbosity=1)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

print(f'Stats - time: {clf.total_time}, iterations: {clf.iterations}, evaluations: {clf.evaluations}')
b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('')
print('c error:', c_error(clf.c_estimate, c))
print('alpha error:', alpha_error(clf.get_STD_alpha(), y))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization import DccpClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error, alpha_error

clf = DccpClassifier(tol=1e-3, tau=1, verbosity=1, dccp_max_iter=100, mosek_max_iter=100, mosek_tol=1e-4)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

print(f'Stats - time: {clf.total_time}, iterations: {clf.iterations}, evaluations: {clf.evaluations}')
b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('')
print('c error:', c_error(clf.c_estimate, c))
print('alpha error:', alpha_error(clf.get_STD_alpha(), y))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization import MMClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error, alpha_error

clf = MMClassifier(verbosity=1, tol=1e-3)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

print(f'Stats - time: {clf.total_time}, iterations: {clf.iterations}, evaluations: {clf.evaluations}')
b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('')
print('c error:', c_error(clf.c_estimate, c))
print('alpha error:', alpha_error(clf.get_STD_alpha(), y))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization.c_estimation import TIcEEstimator

est = TIcEEstimator()
est.fit(X_train, s_train)

from optimization.metrics import c_error, alpha_error
print('c error:', c_error(est.c_estimate, c))
print('alpha error:', alpha_error(est.get_STD_alpha(), y))

# %%
from optimization.c_estimation import ElkanNotoEstimator

est = ElkanNotoEstimator()
est.fit(X_train, s_train)

from optimization.metrics import c_error, alpha_error
print('c error:', c_error(est.c_estimate, c))
print('alpha error:', alpha_error(est.get_STD_alpha(), y))
