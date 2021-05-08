# %%
import numpy as np

import datasets
import matplotlib.pyplot as plt
from data_preprocessing import create_case_control_dataset

error_means_normal = []
error_means_cc = []

c_values = np.arange(0.1, 1, 0.1)

for target_c in c_values:
      X, y = datasets.get_datasets()['credit-a']

      errors_normal = []
      errors_cc = []

      for i in range(10):
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

            from optimization import JointClassifier
            from optimization.functions import oracle_risk, accuracy
            from optimization.metrics import c_error, auc, approximation_error, alpha_error

            clf = JointClassifier()
            clf.fit(X_train, s_train)

            errors_normal.append(alpha_error(clf.get_CC_alpha(), y))

            from optimization.cc_joint_classifier import CcJointClassifier
            from optimization.functions import oracle_risk, accuracy
            from optimization.metrics import c_error, auc, approximation_error

            clf = CcJointClassifier()
            clf.fit(X_train, s_train)

            errors_cc.append(alpha_error(clf.get_CC_alpha(), y))

      error_means_normal.append(np.mean(errors_normal))
      error_means_cc.append(np.mean(errors_cc))

plt.plot(c_values, error_means_normal)
plt.plot(c_values, error_means_cc)
plt.show()
plt.close()
