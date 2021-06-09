# %%
import os

import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from results.test_dicts import marker_styles
from datasets import gen_probit_dataset
from data_preprocessing import preprocess, create_s
from optimization import JointClassifier, CccpClassifier, NaiveClassifier, WeightedClassifier, OracleClassifier, \
    MMClassifier, DccpClassifier
from optimization.c_estimation import TIcEEstimator
from optimization.functions import accuracy, joint_risk, oracle_risk, sigma, add_bias, cccp_risk_wrt_b
from optimization.metrics import approximation_error, auc


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) * 180 / np.pi


n_features = 3
b = np.ones(n_features)
b_bias = np.append([0], b)
Ns = [500, 1000, 2000, 3500, 5000, 7500, 10000]

n_runs = 25
cs = [0.3, 0.6, 0.9]

results = []
for target_c in cs:
    for N in Ns:
        for i in range(n_runs):
            print('')
            print(f'{target_c=}, {N=}, {i+1}/{n_runs}')

            has_res = False

            while not has_res:
                try:
                    X, y = gen_probit_dataset(int(N), b, n_features, include_bias=False)
                    s, c = create_s(y, target_c)

                    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0,
                                                                                   n_best_features=n_features)

                    oracle_clf = OracleClassifier(include_bias=True)
                    oracle_clf.fit(X_train, y_train)
                    oracle_probas = sigma(np.matmul(add_bias(X_train), oracle_clf.params))

                    for clf_name, clf in [
                        ('Oracle', OracleClassifier(include_bias=True)),
                        ('Naive', NaiveClassifier(TIcEEstimator(), include_bias=True)),
                        ('Weighted', WeightedClassifier(TIcEEstimator(), include_bias=True)),
                        ('Joint', JointClassifier(include_bias=True)),
                        ('CCCP', CccpClassifier(include_bias=True, verbosity=0, tol=1e-6, max_iter=40)),
                        ('MM', MMClassifier(include_bias=True, verbosity=0, tol=1e-6, max_iter=40)),
                        ('DCCP', DccpClassifier(tau=1, verbosity=0, tol=1e-3, max_iter=40, mosek_max_iter=1000,
                                                dccp_max_iter=1000)),
                    ]:
                        if clf_name != 'Oracle':
                            clf.fit(X_train, s_train, c=c)
                            error = np.mean(np.abs(b_bias - clf.params))

                            print(' ', clf_name, clf.params)
                            # print('b:', joint_risk(b, X_train, s_train, c))
                            # print("est:", joint_risk(clf.params, X_train, s_train, c))

                            real_probas = sigma(np.matmul(add_bias(X_train), b_bias))
                            est_probas = sigma(np.matmul(add_bias(X_train), clf.params))
                            y_pred = np.where(est_probas >= 0.5, 1, 0)

                            results.append({
                                'c': target_c,
                                'N': N,
                                'Method': clf_name,
                                'Error': error,
                                'RiskB': joint_risk(b_bias, add_bias(X_train), s_train, c),
                                'RiskBHat': joint_risk(clf.params, add_bias(X_train), s_train, clf.c_estimate),
                                'RealProbaError': np.mean(np.abs(est_probas - real_probas)),
                                'OracleProbaError': np.mean(np.abs(est_probas - oracle_probas)),
                                'Angle': angle_between(b_bias[1:], clf.params[1:]),
                                'LengthMultiplier': np.linalg.norm(clf.params[1:]) / np.linalg.norm(b_bias[1:]),
                                'Accuracy': accuracy(y_pred, y_train),
                                'AUC': auc(y_pred, y_train),
                            })
                        elif clf_name == 'Oracle':
                            clf.fit(X_train, y_train)
                            error = np.mean(np.abs(b_bias - clf.params))

                            print(' ', clf_name, clf.params)
                            # print('b:', joint_risk(b, X_train, s_train, c))
                            # print("est:", joint_risk(clf.params, X_train, s_train, c))

                            real_probas = sigma(np.matmul(add_bias(X_train), b_bias))
                            est_probas = sigma(np.matmul(add_bias(X_train), clf.params))
                            y_pred = np.where(est_probas >= 0.5, 1, 0)

                            results.append({
                                'c': target_c,
                                'N': N,
                                'Method': clf_name,
                                'Error': error,
                                'RiskB': joint_risk(b_bias, add_bias(X_train), s_train, c),
                                'RiskBHat': joint_risk(clf.params, add_bias(X_train), s_train, c),
                                'RealProbaError': np.mean(np.abs(est_probas - real_probas)),
                                'OracleProbaError': np.mean(np.abs(est_probas - oracle_probas)),
                                'Angle': angle_between(b_bias[1:], clf.params[1:]),
                                'LengthMultiplier': np.linalg.norm(clf.params[1:]) / np.linalg.norm(b[1:]),
                                'Accuracy': accuracy(y_pred, y_train),
                                'AUC': auc(y_pred, y_train),
                            })
                        # else:
                        #     clf.fit(X_train, s_train, c=c)
                        #     error = np.mean(np.abs(b - clf.params))
                        #
                        #     print(' ', clf_name, clf.params)
                        #     # print('b:', joint_risk(b, X_train, s_train, c))
                        #     # print("est:", joint_risk(clf.params, X_train, s_train, c))
                        #
                        #     real_probas = sigma(np.matmul(X_train, b))
                        #     est_probas = sigma(np.matmul(X_train, clf.params))
                        #     y_pred = np.where(est_probas >= 0.5, 1, 0)
                        #
                        #     results.append({
                        #         'c': target_c,
                        #         'N': N,
                        #         'Method': clf_name,
                        #         'Error': error,
                        #         'RiskB': joint_risk(b, X_train, s_train, c),
                        #         'RiskBHat': joint_risk(clf.params, X_train, s_train, clf.c_estimate),
                        #         'RealProbaError': np.mean(np.abs(est_probas - real_probas)),
                        #         'OracleProbaError': np.mean(np.abs(est_probas - oracle_probas)),
                        #         'Angle': angle_between(b, clf.params),
                        #         'LengthMultiplier': np.linalg.norm(clf.params) / np.linalg.norm(b),
                        #         'Accuracy': accuracy(y_pred, y_train),
                        #         'AUC': auc(y_pred, y_train),
                        #     })
                    has_res = True
                except cvxpy.error.SolverError:
                    print('Solver failed!')
                    pass

    # break

df = pd.DataFrame.from_records(results)

# %%
df.to_csv('probit_res.csv')

# %%
import pandas as pd

df = pd.read_csv('probit_res.csv')

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme()

if not os.path.exists('results/standalone_plots/probit'):
    os.mkdir('results/standalone_plots/probit')
if not os.path.exists('results/standalone_plots/probit/svg'):
    os.mkdir('results/standalone_plots/probit/svg')
from results.test_dicts import marker_styles

def triangle_plots():
    plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 4)

    ax1 = plt.subplot(gs[:2, :2])
    ax2 = plt.subplot(gs[:2, 2:])
    ax3 = plt.subplot(gs[2:4, 1:3])

    return plt.gcf(), [ax1, ax2, ax3]

error_fig, error_axs = triangle_plots()
risk_fig, risk_axs = triangle_plots()
angle_fig, angle_axs = triangle_plots()
real_proba_fig, real_proba_axs = triangle_plots()
oracle_proba_fig, oracle_proba_axs = triangle_plots()
accuracy_fig, accuracy_axs = triangle_plots()
auc_fig, auc_axs = triangle_plots()
length_fig, length_axs = triangle_plots()

for i, target_c in enumerate(df.c.unique()):
    error_ax = error_axs[i]
    angle_ax = angle_axs[i]
    real_proba_ax = real_proba_axs[i]
    oracle_proba_ax = oracle_proba_axs[i]
    accuracy_ax = accuracy_axs[i]
    auc_ax = auc_axs[i]
    length_ax = length_axs[i]

    methods = ['Weighted',
               'Joint',
               'MM',
               'CCCP',
               'DCCP',
               'Oracle']
    for method in methods:
        plot_df = df.loc[(df.c == target_c) & (df.Method == method)]
        plot_df = plot_df.groupby('N').mean().reset_index(drop=False)

        error_ax.plot(plot_df.N, plot_df.Error, **marker_styles[method])
        error_ax.set_xlabel('Rozmiar zbioru danych')
        # error_ax.set_ylabel('$p^{-1} \sum_{i=1}^p |\hat{b_p} - b_p|$')
        error_ax.set_ylabel('PEE')
        error_ax.legend(list(methods))
        error_ax.set_ylim(bottom=-0.1, top=2.1)
        # error_ax.set_title(f'Probit - błąd estymacji parametrów (c={target_c})')
        error_ax.set_title(f'c = {target_c}')

        # risk_ax.plot(plot_df.N, plot_df.RiskB)
        # risk_ax.plot(plot_df.N, plot_df.RiskBHat)
        # risk_ax.set_xlabel('Rozmiar zbioru danych')
        # risk_ax.set_ylabel('Wartość ryzyka')
        # risk_ax.legend(['$\hat{R}_{joint}(b)$ (prawdziwe $b$)', '$\hat{R}_{joint}(\hat{b})$ (estymowane $b$)'])
        # risk_ax.set_title(f'Probit - wartości ryzyka (c={target_c})')
        # risk_ax.set_title(f'c = {target_c}')

        angle_ax.plot(plot_df.N, plot_df.Angle, **marker_styles[method])
        angle_ax.set_xlabel('Rozmiar zbioru danych')
        angle_ax.set_ylabel('Kąt między $\hat{b}$ i $b$ $[^{\circ}]$')
        angle_ax.legend(list(methods))
        angle_ax.set_ylim(bottom=-1, top=11)
        # angle_ax.set_title(f'Probit - kąt (c={target_c})')
        angle_ax.set_title(f'c = {target_c}')

        real_proba_ax.plot(plot_df.N, plot_df.RealProbaError, **marker_styles[method])
        real_proba_ax.set_xlabel('Rozmiar zbioru danych')
        real_proba_ax.set_ylabel('Średni błąd estymacji prawdopodobieństwa')
        real_proba_ax.legend(list(methods))
        real_proba_ax.set_ylim(bottom=-0.01, top=0.15)
        # real_proba_ax.set_title(f'Probit - błąd estymacji rzeczywistego prawdopodobieństwa (c={target_c})')
        real_proba_ax.set_title(f'c = {target_c}')

        oracle_proba_ax.plot(plot_df.N, plot_df.OracleProbaError, **marker_styles[method])
        oracle_proba_ax.set_xlabel('Rozmiar zbioru danych')
        oracle_proba_ax.set_ylabel('AE')
        oracle_proba_ax.legend(list(methods))
        oracle_proba_ax.set_ylim(bottom=-0.01, top=0.15)
        # oracle_proba_ax.set_title(f'Probit - błąd aproksymacji prawdopodobieństwa a posteriori (c={target_c})')
        oracle_proba_ax.set_title(f'c = {target_c}')

        accuracy_ax.plot(plot_df.N, plot_df.Accuracy, **marker_styles[method])
        accuracy_ax.set_xlabel('Rozmiar zbioru danych')
        accuracy_ax.set_ylabel('Accuracy')
        accuracy_ax.legend(list(methods))
        # accuracy_ax.set_title(f'Probit - accuracy (c={target_c})')
        accuracy_ax.set_title(f'c = {target_c}')

        auc_ax.plot(plot_df.N, plot_df.AUC, **marker_styles[method])
        auc_ax.set_xlabel('Rozmiar zbioru danych')
        auc_ax.set_ylabel('AUC')
        auc_ax.legend(list(methods))
        auc_ax.set_ylim(bottom=0.823, top=0.845)
        # auc_ax.set_title(f'Probit - AUC (c={target_c})')
        auc_ax.set_title(f'c = {target_c}')

        length_ax.plot(plot_df.N, plot_df.LengthMultiplier, **marker_styles[method])
        # length_ax.set_yscale('log')
        length_ax.set_xlabel('Rozmiar zbioru danych')
        length_ax.set_ylabel('Wydłużenie wektora parametrów')
        length_ax.legend(list(methods))
        length_ax.set_ylim(bottom=-0.1, top=4)
        # length_ax.set_title(f'Probit - stopień wydłużenia wektora parametrów (c={target_c})')
        length_ax.set_title(f'c = {target_c}')

for fig, name, suptitle in [
    (error_fig, 'ee', 'Probit - błąd estymacji parametrów'),
    (angle_fig, 'angle', 'Probit - kąt (w stopniach) pomiędzy rzeczywistym estymowanym a rzeczywistym wektorem parametrów'),
    (real_proba_fig, 'real_proba', 'Probit - błąd aproksymacji rzeczywistego prawdopodobieństwa '),
    (oracle_proba_fig, 'oracle_proba', 'Probit - błąd aproksymacji prawdopodobieństwa a posteriori'),
    # (accuracy_fig, 'accuracy', 'Probit - accuracy'),
    (auc_fig, 'auc', 'Probit - AUC'),
    (length_fig, 'length', 'Probit - stopień wydłużenia wektora parametrów'),
]:
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(f'results/standalone_plots/probit/probit_{name}_all.png',
                      dpi=300, bbox_inches='tight')
    fig.savefig(f'results/standalone_plots/probit/svg/probit_{name}_all.svg',
                      bbox_inches='tight')
    fig.show()
    plt.close(fig)

plt.close('all')
