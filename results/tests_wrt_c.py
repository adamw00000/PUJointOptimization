import os
import datasets
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from data_preprocessing import create_s, preprocess
from optimization import CccpClassifier, JointClassifier, OracleClassifier, DccpClassifier, \
    NaiveClassifier, MMClassifier, WeightedClassifier
from optimization.c_estimation import TIcEEstimator, ElkanNotoEstimator
from optimization.metrics import approximation_error, c_error, auc, alpha_error

dir_path = os.path.dirname(os.path.realpath(__file__))


def oracle_prediction(X_train, y_train, X_test):
    clf = OracleClassifier()
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)
    return y_proba


def pu_prediction(clf, X_train, s_train, X_test, c=None):
    clf.fit(X_train, s_train, c)

    y_proba = clf.predict_proba(X_test)
    return y_proba


def joint_prediction(clf, X_train, s_train, X_test):
    clf.fit(X_train, s_train)

    y_proba = clf.predict_proba(X_test)
    return y_proba, clf.c_estimate


def calculate_metrics(clf, X_train, y_train, s_train, X_test, y_test, c, oracle_pred, const_c: bool = False):
    if const_c:
        y_pred = pu_prediction(clf, X_train, s_train, X_test, c=c)
        c_estimate = None
    else:
        y_pred, c_estimate = joint_prediction(clf, X_train, s_train, X_test)

    approx_err = approximation_error(y_pred, oracle_pred)
    auc_score = auc(y_test, y_pred)

    if const_c:
        return pd.DataFrame({
            'Metric': ['Błąd aproksymacji (AE) prawdopodobieństwa a posteriori', 'AUC'],
            'Value': [approx_err, auc_score]
        })
    else:
        c_err = c_error(c_estimate, c)
        y = np.concatenate([y_train, y_test])
        alpha_err = alpha_error(clf.get_STD_alpha(), y)

        return pd.DataFrame({
            'Metric': ['Błąd aproksymacji (AE) prawdopodobieństwa a posteriori',
                       r'Błąd estymacji częstości etykietowania',
                       r'Błąd estymacji prawdopodobieństwa a priori',
                       'AUC'],
            'Value': [approx_err, c_err, alpha_err, auc_score]
        })


def get_oracle_metrics(y_test, oracle_pred):
    auc_score = auc(y_test, oracle_pred)

    return pd.DataFrame({
        'Metric': ['AUC'],
        'Value': [auc_score]
    })


def run_test(dataset_name, dataset, target_c, run_number):
    X, y = dataset
    s, c = create_s(y, target_c)
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0.2)

    oracle_pred = oracle_prediction(X_train, y_train, X_test)
    oracle_df = get_oracle_metrics(y_test, oracle_pred)
    oracle_df = oracle_df.assign(Dataset=dataset_name, Method='Oracle', c=target_c)
    oracle_df = pd.concat([
        oracle_df.assign(ConstC=True),
        oracle_df.assign(ConstC=False),
    ])

    dfs = []
    for name in joint_classifiers:
        print(f'--- {dataset_name} ({name}): c = {target_c}, run {run_number + 1}/{total_runs} ---')
        df = calculate_metrics(joint_classifiers[name], X_train, y_train, s_train, X_test, y_test, c, oracle_pred)
        df = df.assign(Dataset=dataset_name, Method=name, c=target_c, RunNumber=run_number, ConstC=False)
        dfs.append(df)
    for name in const_c_classifiers:
        print(f'--- {dataset_name} ({name}): c = {target_c}, run {run_number + 1}/{total_runs} (CONST c) ---')
        df = calculate_metrics(const_c_classifiers[name], X_train, y_train, s_train, X_test, y_test, c, oracle_pred,
                               const_c=True)
        df = df.assign(Dataset=dataset_name, Method=name, c=target_c, RunNumber=run_number, ConstC=True)
        dfs.append(df)
    return pd.concat(dfs), oracle_df


def plot_metrics(metrics_df):
    mean_metrics_df = metrics_df.groupby(['Dataset', 'ConstC', 'Method', 'c', 'Metric']) \
        .Value \
        .mean() \
        .reset_index(drop=False)

    split_dataset_dict = dict(tuple(mean_metrics_df.groupby('Dataset')))
    for dataset_name in split_dataset_dict:
        split_const_c_dict = dict(tuple(split_dataset_dict[dataset_name].groupby('ConstC')))

        for const_c in split_const_c_dict:
            split_metric_dict = dict(tuple(split_const_c_dict[const_c].groupby('Metric')))

            for metric in split_metric_dict:
                plt.figure(figsize=(8, 6))
                sns.set_theme()
                ax = plt.gca()

                metric_df = split_metric_dict[metric]
                split_method_dict = dict(tuple(metric_df.groupby('Method')))

                for method in split_method_dict:
                    method_df = split_method_dict[method]

                    ax.plot(method_df.c, method_df.Value)
                    ax.scatter(method_df.c, method_df.Value)

                const_c_string = f'znane c' if const_c else f'estymowane c'

                plt.legend([name for name in split_method_dict])
                plt.xlabel(r'Częstość etykietowania $c$')
                plt.ylabel(metric)
                plt.title(f'{dataset_name} - {metric} - {const_c_string}')
                plt.savefig(os.path.join('plots',
                                         f'{dataset_name} - {metric} - {const_c_string}.png'),
                            dpi=150, bbox_inches='tight')
                plt.show()


def get_latex_table(metric, metric_pivot, rank_pivot):
    best_function = {
        'Błąd aproksymacji (AE) prawdopodobieństwa a posteriori': np.min,
        r'Błąd estymacji częstości etykietowania': np.min,
        r'Błąd estymacji prawdopodobieństwa a priori': np.min,
        'AUC': np.max
    }

    metric_pivot = metric_pivot.reset_index()
    rank_pivot = rank_pivot.reset_index()

    for const_c in [True, False]:
        latex_string = f"{'Dataset':16} "
        for col in metric_pivot.columns:
            if col in ['ConstC', 'index', 'Dataset']:
                continue
            if np.sum(np.isnan(metric_pivot.loc[metric_pivot.ConstC == const_c, col])):
                continue
            latex_string += f"& {col:14} "
        latex_string += '\\\\\n'

        for row in metric_pivot.loc[metric_pivot.ConstC == const_c].iterrows():
            latex_string += f"{row[1]['Dataset']:16} "

            values = [item[1] for (i, item) in enumerate(row[1].items())
                             if i >= 2 and not np.isnan(item[1])]
            non_oracle_values = [item[1] for (i, item) in enumerate(row[1].items())
                             if i >= 2 and not np.isnan(item[1]) and not item[0] == 'Oracle']

            for value in values:
                text = f"{value}"
                if value == best_function[metric](non_oracle_values):
                    text = "\\textbf{" + text + "}"
                latex_string += f"& {text:14} "
            latex_string += '\\\\\n'

        for row in rank_pivot.loc[rank_pivot.ConstC == const_c].iterrows():
            latex_string += f"{'Rank':16} "
            values = [item[1] for (i, item) in enumerate(row[1].items())
                      if i >= 1 and not np.isnan(item[1])]
            non_oracle_values = [item[1] for (i, item) in enumerate(row[1].items())
                      if i >= 1 and not np.isnan(item[1]) and not item[0] == 'Oracle']
            for value in values:
                text = f"{value}"
                if value == np.min(non_oracle_values):
                    text = "\\textbf{" + text + "}"
                latex_string += f"& {text:14} "
            latex_string += '\\\\\n'

        with open(os.path.join('latex',
                               f'{metric}_latex_{"known" if const_c else "estimated"}_c.txt'),
                  'w') as f:
            f.write(latex_string)


def create_rankings(metrics_df, oracle_metrics_df):
    is_metric_increasing = {
        'Błąd aproksymacji (AE) prawdopodobieństwa a posteriori': True,
        r'Błąd estymacji częstości etykietowania': True,
        r'Błąd estymacji prawdopodobieństwa a priori': True,
        'AUC': False
    }

    for metric in metrics_df.Metric.unique():
        df = metrics_df.loc[metrics_df.Metric == metric]
        oracle_df = oracle_metrics_df.loc[oracle_metrics_df.Metric == metric]

        mean_metrics_df = df.groupby(['Dataset', 'ConstC', 'Method']) \
            .Value \
            .mean() \
            .reset_index(drop=False)
        mean_oracle_metrics_df = oracle_df.groupby(['Dataset', 'ConstC', 'Method']) \
            .Value \
            .mean() \
            .reset_index(drop=False)

        mean_metrics_df = pd.concat([mean_oracle_metrics_df, mean_metrics_df])
        mean_metrics_df.Value = mean_metrics_df.Value.round(3)

        ranks = mean_metrics_df.groupby(['Dataset', 'ConstC'])\
            .Value\
            .rank(ascending=is_metric_increasing[metric])
        ranked_mean_metrics_df = mean_metrics_df.assign(Rank=ranks)
        mean_rank = ranked_mean_metrics_df\
            .groupby(['ConstC', 'Method'])\
            .Rank\
            .mean()\
            .reset_index()

        metric_pivot = pd.pivot_table(mean_metrics_df, values='Value',
                                      index=['ConstC', 'Dataset'],
                                      columns=['Method'])\
            .round(3)
        rank_pivot = pd.pivot_table(mean_rank, values='Rank',
                                    index=['ConstC'], columns=['Method'])\
            .round(3)

        metric_pivot.to_csv(os.path.join('csv', f'mean_metrics_by_dataset_{metric}.csv'))
        rank_pivot.to_csv(os.path.join('csv', f'mean_ranks_{metric}.csv'))

        get_latex_table(metric, metric_pivot, rank_pivot)


if __name__ == '__main__':
    if not os.path.exists('plots'):
        os.mkdir('plots')
    if not os.path.exists('csv'):
        os.mkdir('csv')
    if not os.path.exists('latex'):
        os.mkdir('latex')

    data = datasets.get_datasets()

    const_c_classifiers = {
        'Naive': NaiveClassifier(TIcEEstimator()),
        'Weighted': WeightedClassifier(TIcEEstimator()),
        'Joint': JointClassifier(),
        'CCCP': CccpClassifier(verbosity=1),
        'MM': MMClassifier(verbosity=1),
        # 'DCCP': DccpClassifier(tau=1, verbosity=1),
    }

    joint_classifiers = {
        'Naive - TIcE': NaiveClassifier(TIcEEstimator()),
        'Naive - EN': NaiveClassifier(ElkanNotoEstimator()),
        'Weighted - TIcE': WeightedClassifier(TIcEEstimator()),
        'Weighted - EN': WeightedClassifier(ElkanNotoEstimator()),
        'Joint': JointClassifier(),
        'CCCP': CccpClassifier(verbosity=1),
        'MM': MMClassifier(verbosity=1),
        # 'DCCP': DccpClassifier(tau=1, verbosity=1),
    }

    total_runs = 100
    c_values = np.arange(0.1, 1, 0.1)

    num_cores = multiprocessing.cpu_count() - 1
    results = Parallel(n_jobs=num_cores)(delayed(run_test)(dataset_name, data[dataset_name], c, run_number)
                                         for dataset_name, c, run_number in zip(
                                             np.repeat(np.repeat(list(data.keys()), total_runs), len(c_values)),
                                             np.tile(np.repeat(c_values, total_runs), len(data)),
                                             np.tile(np.tile(range(total_runs), len(c_values)), len(data))
                                         ))

    metrics_dfs = [res[0] for res in results]
    oracle_dfs = [res[1] for res in results]

    metrics_df = pd.concat(metrics_dfs)
    metrics_df.to_csv(os.path.join('csv', 'raw_results.csv'))
    oracle_df = pd.concat(oracle_dfs)
    oracle_df.to_csv(os.path.join('csv', 'raw_oracle_results.csv'))

    metrics_df = pd.read_csv(os.path.join('csv', 'raw_results.csv'))
    oracle_df = pd.read_csv(os.path.join('csv', 'raw_oracle_results.csv'))
    plot_metrics(metrics_df)
    create_rankings(metrics_df, oracle_df)
