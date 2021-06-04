# %%
import multiprocessing

import numpy as np
from joblib import Parallel, delayed

import datasets
from data_preprocessing import create_case_control_dataset, preprocess
from optimization import JointClassifier, MMClassifier
from optimization.c_estimation import TIcEEstimator
import pandas as pd

c_values = np.arange(0.1, 1, 0.1)
n_repeats = 100

dataset = 'credit-a'
# dataset = 'spambase'

def run(target_c, i):
    print(f'c = {target_c}, fit {i + 1}/{n_repeats}')
    # X, y = datasets.get_datasets()['credit-a']
    X, y = datasets.get_datasets()[dataset]

    X_new, y_new, s, c = create_case_control_dataset(X, y, target_c)
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X_new, y_new, s, test_size=0.2)

    # joint_df = pd.DataFrame()
    # tice_df = pd.DataFrame()

    clf = JointClassifier()
    clf.fit(X_train, s_train)
    joint_df = pd.DataFrame({
        'target_c': [target_c],
        'c_estimate': [clf.c_estimate]
    })

    clf = TIcEEstimator()
    clf.fit(X_train, s_train)
    tice_df = pd.DataFrame({
        'target_c': [target_c],
        'c_estimate': [clf.c_estimate]
    })

    # clf = CccpClassifier(max_iter=20)
    clf = MMClassifier(tol=1e-4, max_iter=20)
    clf.fit(X_train, s_train)
    split_df = pd.DataFrame({
        'target_c': [target_c],
        'c_estimate': [clf.c_estimate]
    })

    return joint_df, tice_df, split_df

num_cores = multiprocessing.cpu_count() - 1
results = Parallel(n_jobs=num_cores)(delayed(run)(c, i)
                                     for c, i in zip(
                                         np.repeat(list(c_values), n_repeats),
                                         np.tile(range(n_repeats), len(c_values)),
                                     ))

joint_dfs = [x[0] for x in results]
tice_dfs = [x[1] for x in results]
split_dfs = [x[2] for x in results]

# %%
joint_df = pd.concat(joint_dfs)
joint_df.target_c = np.round(joint_df.target_c, 1)
# joint_df.to_csv()
tice_df = pd.concat(tice_dfs)
tice_df.target_c = np.round(tice_df.target_c, 1)
# tice_df.to_csv()
split_df = pd.concat(split_dfs)
split_df.target_c = np.round(split_df.target_c, 1)
# cccp_df.to_csv()

# %%
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme()
plt.figure(figsize=(10, 8))
gs = gridspec.GridSpec(4, 4)

ax1 = plt.subplot(gs[:2, :2])
ax2 = plt.subplot(gs[:2, 2:])
ax3 = plt.subplot(gs[2:4, 1:3])

sns.stripplot(ax=ax1, data=tice_df, x='target_c', y='c_estimate', size=2)
sns.stripplot(ax=ax2, data=joint_df, x='target_c', y='c_estimate', size=2)
sns.stripplot(ax=ax3, data=split_df, x='target_c', y='c_estimate', size=2)

ticks = np.arange(0, 1.1, 0.1)

ax1.set_title('TIcE')
ax1.set_xlabel('Prawdziwe c')
ax1.set_ylabel('Estymowane c')
ax1.set_yticks(ticks)
ax2.set_title('Joint')
ax2.set_xlabel('Prawdziwe c')
ax2.set_ylabel('Estymowane c')
ax2.set_yticks(ticks)
ax3.set_title('Split optimization (MM)')
ax3.set_xlabel('Prawdziwe c')
ax3.set_ylabel('Estymowane c')
ax3.set_yticks(ticks)

plt.suptitle(f'Estymacja c - {dataset}')
plt.gcf().tight_layout()

plt.savefig(os.path.join('results',
                         'standalone_plots',
                         f'c_real_vs_estimate_{dataset}.png'),
            dpi=300, bbox_inches='tight')
plt.savefig(os.path.join('results',
                         'standalone_plots',
                         f'c_real_vs_estimate_{dataset}.svg'),
            bbox_inches='tight')
plt.show()
plt.close()
