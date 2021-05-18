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

                    oracle_clf = OracleClassifier(include_bias=False)
                    oracle_clf.fit(X_train, y_train)
                    oracle_probas = sigma(np.matmul(X_train, oracle_clf.params))

                    for clf_name, clf in [
                        ('Naive', NaiveClassifier(TIcEEstimator(), include_bias=False)),
                        ('Weighted', WeightedClassifier(TIcEEstimator(), include_bias=False)),
                        ('Joint', JointClassifier(include_bias=False)),
                        ('CCCP', CccpClassifier(include_bias=False, verbosity=0, tol=1e-6, max_iter=40)),
                        ('MM', MMClassifier(include_bias=False, verbosity=0, tol=1e-6, max_iter=40)),
                        ('DCCP', DccpClassifier(tau=1, verbosity=0, tol=1e-3, max_iter=40, mosek_max_iter=1000, dccp_max_iter=1000)),
                    ]:
                        clf.fit(X_train, s_train, c=c)

                        if clf_name != 'DCCP':
                            error = np.mean(np.abs(b - clf.params))

                            print(' ', clf_name, clf.params)
                            # print('b:', joint_risk(b, X_train, s_train, c))
                            # print("est:", joint_risk(clf.params, X_train, s_train, c))

                            real_probas = sigma(np.matmul(X_train, b))
                            est_probas = sigma(np.matmul(X_train, clf.params))
                            y_pred = np.where(est_probas >= 0.5, 1, 0)

                            results.append({
                                'c': target_c,
                                'N': N,
                                'Method': clf_name,
                                'Error': error,
                                'RiskB': joint_risk(b, X_train, s_train, c),
                                'RiskBHat': joint_risk(clf.params, X_train, s_train, clf.c_estimate),
                                'RealProbaError': np.mean(np.abs(est_probas - real_probas)),
                                'OracleProbaError': np.mean(np.abs(est_probas - oracle_probas)),
                                'Angle': angle_between(b, clf.params),
                                'Accuracy': accuracy(y_pred, y_train),
                                'AUC': auc(y_pred, y_train),
                            })
                        else:
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
                                'Accuracy': accuracy(y_pred, y_train),
                                'AUC': auc(y_pred, y_train),
                            })
                    has_res = True
                except cvxpy.error.SolverError:
                    print('Solver failed!')
                    pass

    # break

df = pd.DataFrame.from_records(results)

# %%
sns.set_theme()

if not os.path.exists('results/standalone_plots/probit'):
    os.mkdir('results/standalone_plots/probit')

for target_c in df.c.unique():
    error_fig, error_ax = plt.subplots()
    risk_fig, risk_ax = plt.subplots()
    angle_fig, angle_ax = plt.subplots()
    real_proba_fig, real_proba_ax = plt.subplots()
    oracle_proba_fig, oracle_proba_ax = plt.subplots()
    accuracy_fig, accuracy_ax = plt.subplots()
    auc_fig, auc_ax = plt.subplots()

    methods = df.Method.unique()
    for method in methods:
        plot_df = df.loc[(df.c == target_c) & (df.Method == method)]
        plot_df = plot_df.groupby('N').mean().reset_index(drop=False)

        error_ax.plot(plot_df.N, plot_df.Error, **marker_styles[method])
        error_ax.set_xlabel('Rozmiar zbioru danych')
        error_ax.set_ylabel('$p^{-1} \sum_{i=1}^p |\hat{b_p} - b_p|$')
        error_ax.legend(list(methods))
        error_ax.set_title(f'Probit - błąd estymacji parametrów (c={target_c})')

        # risk_ax.plot(plot_df.N, plot_df.RiskB)
        # risk_ax.plot(plot_df.N, plot_df.RiskBHat)
        # risk_ax.set_xlabel('Rozmiar zbioru danych')
        # risk_ax.set_ylabel('Wartość ryzyka')
        # risk_ax.legend(['$\hat{R}_{joint}(b)$ (prawdziwe $b$)', '$\hat{R}_{joint}(\hat{b})$ (estymowane $b$)'])
        # risk_ax.set_title(f'Probit - wartości ryzyka (c={target_c})')

        angle_ax.plot(plot_df.N, plot_df.Angle, **marker_styles[method])
        angle_ax.set_xlabel('Rozmiar zbioru danych')
        angle_ax.set_ylabel('Kąt między $\hat{b}$ i $b$')
        angle_ax.legend(list(methods))
        angle_ax.set_title(f'Probit - kąt (c={target_c})')

        real_proba_ax.plot(plot_df.N, plot_df.RealProbaError, **marker_styles[method])
        real_proba_ax.set_xlabel('Rozmiar zbioru danych')
        real_proba_ax.set_ylabel('Średni błąd estymacji prawdopodobieństwa')
        real_proba_ax.legend(list(methods))
        real_proba_ax.set_title(f'Probit - błąd estymacji prawdopodobieństwa (c={target_c})')

        oracle_proba_ax.plot(plot_df.N, plot_df.OracleProbaError, **marker_styles[method])
        oracle_proba_ax.set_xlabel('Rozmiar zbioru danych')
        oracle_proba_ax.set_ylabel('Średni błąd estymacji prawdopodobieństwa')
        oracle_proba_ax.legend(list(methods))
        oracle_proba_ax.set_title(f'Probit - błąd estymacji prawdopodobieństwa wzgl. oracle (c={target_c})')

        accuracy_ax.plot(plot_df.N, plot_df.Accuracy, **marker_styles[method])
        accuracy_ax.set_xlabel('Rozmiar zbioru danych')
        accuracy_ax.set_ylabel('Accuracy')
        accuracy_ax.legend(list(methods))
        accuracy_ax.set_title(f'Probit - accuracy (c={target_c})')

        auc_ax.plot(plot_df.N, plot_df.AUC, **marker_styles[method])
        auc_ax.set_xlabel('Rozmiar zbioru danych')
        auc_ax.set_ylabel('AUC')
        auc_ax.legend(list(methods))
        auc_ax.set_title(f'Probit - AUC (c={target_c})')

    error_fig.tight_layout()
    error_fig.savefig(f'results/standalone_plots/probit/probit_ee_{target_c}_all.png',
                      dpi=300, bbox_inches='tight')
    error_fig.show()
    plt.close(error_fig)

    # risk_fig.tight_layout()
    # risk_fig.savefig(f'results/standalone_plots/probit/probit_risk_{target_c}_all.png',
    #                  dpi=300, bbox_inches='tight')
    # risk_fig.show()
    # plt.close(risk_fig)

    angle_fig.tight_layout()
    angle_fig.savefig(f'results/standalone_plots/probit/probit_angle_{target_c}_all.png',
                      dpi=300, bbox_inches='tight')
    angle_fig.show()
    plt.close(angle_fig)

    real_proba_fig.tight_layout()
    real_proba_fig.savefig(f'results/standalone_plots/probit/probit_real_proba_{target_c}_all.png',
                      dpi=300, bbox_inches='tight')
    real_proba_fig.show()
    plt.close(real_proba_fig)

    oracle_proba_fig.tight_layout()
    oracle_proba_fig.savefig(f'results/standalone_plots/probit/probit_oracle_proba_{target_c}_all.png',
                      dpi=300, bbox_inches='tight')
    oracle_proba_fig.show()
    plt.close(oracle_proba_fig)

    accuracy_fig.tight_layout()
    accuracy_fig.savefig(f'results/standalone_plots/probit/probit_accuracy_{target_c}_all.png',
                      dpi=300, bbox_inches='tight')
    accuracy_fig.show()
    plt.close(accuracy_fig)

    auc_fig.tight_layout()
    auc_fig.savefig(f'results/standalone_plots/probit/probit_AUC_{target_c}_all.png',
                      dpi=300, bbox_inches='tight')
    auc_fig.show()
    plt.close(auc_fig)
