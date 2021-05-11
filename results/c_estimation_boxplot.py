# %%
import numpy as np

import datasets
from data_preprocessing import create_case_control_dataset, preprocess
from optimization import JointClassifier
from optimization.c_estimation import TIcEEstimator
import pandas as pd

c_values = np.arange(0.1, 1, 0.1)
n_repeats = 100

dataset = 'credit-a'
# dataset = 'spambase'

oracle_dfs = []
tice_dfs = []
for target_c in c_values:
      for i in range(n_repeats):
            print(f'c = {target_c}, fit {i+1}/{n_repeats}')
            # X, y = datasets.get_datasets()['credit-a']
            X, y = datasets.get_datasets()[dataset]

            X_new, y_new, s, c = create_case_control_dataset(X, y, target_c)
            X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X_new, y_new, s, test_size=0.2)

            clf = JointClassifier()
            clf.fit(X_train, s_train)
            df = pd.DataFrame({
                  'target_c': [target_c],
                  'c_estimate': [clf.c_estimate]
            })
            oracle_dfs.append(df)

            clf = TIcEEstimator()
            clf.fit(X_train, s_train)
            df = pd.DataFrame({
                  'target_c': [target_c],
                  'c_estimate': [clf.c_estimate]
            })
            tice_dfs.append(df)

joint_df = pd.concat(oracle_dfs)
joint_df.target_c = np.round(joint_df.target_c, 1)
# joint_df.to_csv()
tice_df = pd.concat(tice_dfs)
tice_df.target_c = np.round(tice_df.target_c, 1)
# tice_df.to_csv()

# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure()
sns.stripplot(data=joint_df, x='target_c', y='c_estimate', size=3)
ticks = np.arange(0, 1.1, 0.1)
plt.yticks(ticks)
plt.xlabel('Prawdziwe c')
plt.ylabel('Estymowane c')

plt.savefig(os.path.join('results',
                         'standalone_plots',
                         f'c_real_vs_estimate_{dataset}_joint.png'),
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure()
sns.boxplot(data=joint_df, x='target_c', y='c_estimate')
ticks = np.arange(0, 1.1, 0.1)
plt.yticks(ticks)
plt.xlabel('Prawdziwe c')
plt.ylabel('Estymowane c')

plt.savefig(os.path.join('results',
                         'standalone_plots',
                         f'c_real_vs_estimate_{dataset}_joint_boxplot.png'),
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure()
sns.stripplot(data=tice_df, x='target_c', y='c_estimate', size=3)
ticks = np.arange(0, 1.1, 0.1)
plt.yticks(ticks)
plt.xlabel('Prawdziwe c')
plt.ylabel('Estymowane c')

plt.savefig(os.path.join('results',
                         'standalone_plots',
                         f'c_real_vs_estimate_{dataset}_ward.png'),
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# %%
import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()
plt.figure()
sns.boxplot(data=tice_df, x='target_c', y='c_estimate')
ticks = np.arange(0, 1.1, 0.1)
plt.yticks(ticks)
plt.xlabel('Prawdziwe c')
plt.ylabel('Estymowane c')

plt.savefig(os.path.join('results',
                         'standalone_plots',
                         f'c_real_vs_estimate_{dataset}_ward_boxplot.png'),
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()
