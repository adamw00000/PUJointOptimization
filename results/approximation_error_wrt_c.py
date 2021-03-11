import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_preprocessing import create_s, preprocess
from optimization import CccpClassifier, JointClassifier, OracleClassifier
from optimization.metrics import approximation_error, c_error, auc


def oracle_prediction(X_train, y_train, X_test):
    clf = OracleClassifier()
    clf.fit(X_train, y_train.to_numpy())

    y_proba = clf.predict_proba(X_test)
    return y_proba


def pu_prediction(clf, X_train, s_train, X_test):
    clf.fit(X_train, s_train)

    y_proba = clf.predict_proba(X_test)
    return y_proba, clf.c_estimate


def calculate_metrics(clf, X_train, s_train, X_test, y_test, c):
    y_pred, c_estimate = pu_prediction(clf, X_train, s_train, X_test)

    approx_err = approximation_error(y_pred, oracle_pred)
    c_err = c_error(c, c_estimate)
    auc_score = auc(y_test, y_pred)

    return pd.DataFrame({
        'Metric': ['Approximation error (AE) for posterior', r'Label frequency error', 'AUC'],
        'Value': [approx_err, c_err, auc_score]
    })


if __name__ == '__main__':
    dataset_name = 'spambase'
    X, y = datasets.load_spambase()

    classifiers = {
        'Joint': JointClassifier(),
        'CCCP': CccpClassifier(verbosity=1)
    }

    result_dfs = []
    c_values = np.arange(0.1, 1, 0.1)
    for c in c_values:
        errors = {name: [] for name in classifiers}

        for run_number in range(10):
            s = create_s(y, c)
            X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0.2)

            oracle_pred = oracle_prediction(X_train, y_train, X_test)

            for name in classifiers:
                print(f'--- {name}: c = {c} ---')
                df = calculate_metrics(classifiers[name], X_train, s_train, X_test, y_test, c)
                df = df.assign(Method=name, c=c, RunNumber=run_number)
                result_dfs.append(df)

    metrics_df = pd.concat(result_dfs)
    mean_metrics_df = metrics_df.groupby(['Method', 'c', 'Metric'])\
        .Value\
        .mean()\
        .reset_index(drop=False)

    split_metric_dict = dict(tuple(mean_metrics_df.groupby('Metric')))
    for metric in split_metric_dict:
        fig = plt.figure()
        ax = plt.gca()

        metric_df = split_metric_dict[metric]
        split_method_dict = dict(tuple(metric_df.groupby('Method')))

        for method in split_method_dict:
            method_df = split_method_dict[method]

            ax.plot(method_df.c, method_df.Value)
            ax.scatter(method_df.c, method_df.Value)

        plt.legend([name for name in split_method_dict])
        plt.xlabel(r'Label frequency $c$')
        plt.ylabel(metric)
        plt.savefig(f'{dataset_name} - {metric}.png', dpi=150, bbox_inches='tight')
        plt.show()


