# %%
import os
import matplotlib.pyplot as plt
import numpy as np

import datasets
from data_preprocessing import preprocess, create_s

from optimization import JointClassifier, CccpClassifier, MMClassifier, DccpClassifier
from optimization.functions import oracle_risk, joint_risk

fig, axs = plt.subplots(2, 3, figsize=(16, 12))


def get_risk_values(clf, X_test, y_test, target_c):
    v = []

    for b in clf.param_history:
        # v.append(oracle_risk(b, X_test, y_test))
        v.append(joint_risk(b, X_test, y_test, target_c))

    # return v
    return clf.risk_values


for i, dataset in enumerate([datasets.gen_M1_dataset(), datasets.gen_M2_dataset()]):
    for j, target_c in enumerate([.3, .5, .7]):
        print(f'Model {i+1}, c: {target_c}')
        X, y = datasets.gen_M2_dataset()
        s, c = create_s(y, target_c)

        ax = axs[i, j]

        X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0.2)

        clf = JointClassifier(tol=1e-10, max_iter=100, get_info=True)
        clf.fit(X_train, s_train)

        v = get_risk_values(clf, X_test, y_test, target_c)
        ax.plot(np.array(v[:100]) * -5000)
        print('Joint:', v[:100][-1] * -5000)

        clf = CccpClassifier(verbosity=0, tol=1e-10, cccp_max_iter=20, max_iter=10, cg_max_iter=10,
                             get_info=True, reset_params_each_iter=False)
        clf.fit(X_train, s_train)

        v = get_risk_values(clf, X_test, y_test, target_c)
        ax.plot(np.array(v[:100]) * -5000)
        print('CCCP:', v[:100][-1] * -5000)

        clf = MMClassifier(verbosity=0, tol=1e-10, mm_max_iter=20, max_iter=10, osqp_max_iter=10,
                           get_info=True, reset_params_each_iter=False)
        clf.fit(X_train, s_train)

        v = get_risk_values(clf, X_test, y_test, target_c)
        ax.plot((np.array(range(11))) * 10, np.array(v[:11]) * -5000)
        print('MM:', v[:100][-1] * -5000)

        clf = DccpClassifier(tau=1, verbosity=0, tol=1e-10, dccp_max_iter=2, max_iter=10, mosek_max_iter=100,
                             get_info=True, reset_params_each_iter=False)
        clf.fit(X_train, s_train)

        v = get_risk_values(clf, X_test, y_test, target_c)
        ax.plot((np.array(range(11))) * 10, np.array(v[:11]) * -5000)
        print('DCCP:', v[:100][-1] * -5000)

        # ax.legend(['Joint', 'CCCP', 'MM'])
        ax.legend(['Joint', 'CCCP', 'MM', 'DCCP'])
        ax.set_title(f'Model {i+1}, c: {target_c}')

# for ax in axs.flat:
#     ax.label_outer()

if not os.path.exists('results/standalone_plots'):
    os.mkdir('results/standalone_plots')

fig.suptitle('Zbieżność metod', fontsize=20)
plt.savefig('results/standalone_plots/convergence.png', dpi=150)
plt.show()

# %%
