import datasets
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from data_preprocessing import create_s, preprocess
from optimization import CccpClassifier, JointClassifier, OracleClassifier, DccpClassifier, NaiveClassifier
from optimization.metrics import approximation_error, c_error, auc


def oracle_prediction(X_train, y_train, X_test):
    clf = OracleClassifier()
    clf.fit(X_train, y_train.to_numpy())

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


def calculate_metrics(clf, X_train, s_train, X_test, y_test, c, oracle_pred, const_c: bool = False):
    if const_c:
        y_pred = pu_prediction(clf, X_train, s_train, X_test, c=c)
        c_estimate = None
    else:
        y_pred, c_estimate = joint_prediction(clf, X_train, s_train, X_test)

    approx_err = approximation_error(y_pred, oracle_pred)
    auc_score = auc(y_test, y_pred)

    if const_c:
        return pd.DataFrame({
            'Metric': ['Approximation error (AE) for posterior', 'AUC'],
            'Value': [approx_err, auc_score]
        })
    else:
        c_err = c_error(c, c_estimate)

        return pd.DataFrame({
            'Metric': ['Approximation error (AE) for posterior', r'Label frequency error', 'AUC'],
            'Value': [approx_err, c_err, auc_score]
        })


def run_test(dataset_name, dataset, target_c, run_number):
    X, y = dataset
    s, c = create_s(y, target_c)
    X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0.2)

    oracle_pred = oracle_prediction(X_train, y_train, X_test)

    dfs = []
    for name in joint_classifiers:
        print(f'--- {dataset_name} ({name}): c = {target_c}, run {run_number + 1}/{total_runs} ---')
        df = calculate_metrics(joint_classifiers[name], X_train, s_train, X_test, y_test, c, oracle_pred)
        df = df.assign(Dataset=dataset_name, Method=name, c=target_c, RunNumber=run_number, ConstC=False)
        dfs.append(df)
    for name in const_c_classifiers:
        print(f'--- {dataset_name} ({name}): c = {target_c}, run {run_number + 1}/{total_runs} (CONST c) ---')
        df = calculate_metrics(const_c_classifiers[name], X_train, s_train, X_test, y_test, c, oracle_pred,
                               const_c=True)
        df = df.assign(Dataset=dataset_name, Method=name, c=target_c, RunNumber=run_number, ConstC=True)
        dfs.append(df)
    return pd.concat(dfs)


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
                plt.figure()
                sns.set_theme()
                ax = plt.gca()

                metric_df = split_metric_dict[metric]
                split_method_dict = dict(tuple(metric_df.groupby('Method')))

                for method in split_method_dict:
                    method_df = split_method_dict[method]

                    ax.plot(method_df.c, method_df.Value)
                    ax.scatter(method_df.c, method_df.Value)

                const_c_string = f'constant c' if const_c else f'estimated c'

                plt.legend([name for name in split_method_dict])
                plt.xlabel(r'Label frequency $c$')
                plt.ylabel(metric)
                plt.title(f'{dataset_name} - {metric} - {const_c_string}')
                plt.savefig(f'{dataset_name} - {metric} - {const_c_string}.png', dpi=150, bbox_inches='tight')
                plt.show()


if __name__ == '__main__':
    data = datasets.get_datasets()

    const_c_classifiers = {
        'Naive': NaiveClassifier(),
        'Joint': JointClassifier(),
        'CCCP': CccpClassifier(verbosity=1),
        # 'DCCP': DccpClassifier(tau=1, verbosity=1),
    }

    joint_classifiers = {
        'Joint': JointClassifier(),
        'CCCP': CccpClassifier(verbosity=1),
        # 'DCCP': DccpClassifier(tau=1, verbosity=1),
    }

    total_runs = 100
    c_values = np.arange(0.1, 1, 0.1)

    num_cores = multiprocessing.cpu_count() - 1
    result_dfs = Parallel(n_jobs=num_cores)(delayed(run_test)(dataset_name, data[dataset_name], c, run_number)
                                            for dataset_name, c, run_number in zip(
                                                np.repeat(np.repeat(list(data.keys()), total_runs), len(c_values)),
                                                np.tile(np.repeat(c_values, total_runs), len(data)),
                                                np.tile(np.tile(range(total_runs), len(c_values)), len(data))
                                            ))

    metrics_df = pd.concat(result_dfs)
    plot_metrics(metrics_df)
