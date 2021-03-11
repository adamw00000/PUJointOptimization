# %%
import datasets
from data_preprocessing import create_s

c = 0.1
X, y = datasets.load_spambase()
s = create_s(y, c)

# %%
from data_preprocessing import preprocess

X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size = 0.2)

# %%
from optimization import OracleClassifier
from optimization.functions import oracle_risk, accuracy

clf = OracleClassifier()
clf.fit(X_train, y_train.to_numpy())

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

y_proba_oracle = y_proba

# %%
import numpy as np

from optimization import NaiveClassifier
from optimization.functions import oracle_risk, accuracy

clf = NaiveClassifier(c=c)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)

# %%
import numpy as np

from optimization import JointClassifier
from optimization.functions import oracle_risk, accuracy

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

estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)

# %%
import numpy as np

from optimization import CccpClassifier
from optimization.functions import oracle_risk, accuracy

clf = CccpClassifier()
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)

# %%
import numpy as np

from optimization import DccpClassifier
from optimization.functions import oracle_risk, accuracy

clf = DccpClassifier(c=c, tau=1)
clf.fit(X_train, s_train)

y_proba = clf.predict_proba(X_test)
y_pred = clf.predict(X_test)

b = clf.get_params()
risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)