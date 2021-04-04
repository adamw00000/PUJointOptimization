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
clf.fit(X_train, y_train.to_numpy())

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
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import auc, approximation_error

clf = NaiveClassifier()
clf.fit(X_train, s_train, c)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

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
from optimization import CccpClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error

clf = CccpClassifier(verbosity=1)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('c error:', c_error(clf.c_estimate, c))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization import DccpClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error

clf = DccpClassifier(tau=1, verbosity=1, dccp_max_iter=1000)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('c error:', c_error(clf.c_estimate, c))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))

# %%
from optimization import MMClassifier
from optimization.functions import oracle_risk, accuracy
from optimization.metrics import c_error, auc, approximation_error

clf = MMClassifier(verbosity=1)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

print('c error:', c_error(clf.c_estimate, c))
print('AUC:', auc(y_test, y_pred))
print('Approximation error:', approximation_error(y_proba, y_proba_oracle))