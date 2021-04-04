import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif


def preprocess(X, y, s, test_size=0.2, n_best_features=5):
    X = X.fillna(X.mean())

    mutual_info_scores = mutual_info_classif(X, s)
    mutual_info_best = np.argsort(mutual_info_scores)[::-1]
    X_filtered = X.iloc[:, mutual_info_best[:n_best_features]]

    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X_filtered, y, s, test_size=test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, s_train, s_test


def create_s(y, c):   # c - label_frequency
    s = np.array(y)
    positives = np.where(y == 1)[0]
    n_y1 = len(positives)

    new_unlabeled_samples = positives[np.random.random(len(positives)) < 1 - c]
    s[new_unlabeled_samples] = 0
    n_s0 = len(new_unlabeled_samples)

    n_s1 = n_y1 - n_s0
    real_c = n_s1 / n_y1
    return s, real_c


def create_case_control_dataset(X, y, c):   # c - label_frequency
    positives = np.where(y == 1)[0]
    negatives = np.where(y == 0)[0]

    new_labeled_samples = positives[np.random.random(len(positives)) < c]
    X_pos = X.iloc[positives, :]
    y_pos = y[positives]
    s_pos = np.zeros(len(positives))
    s_pos[new_labeled_samples] = 1

    new_unlabeled_samples = negatives[np.random.random(len(negatives)) < 1 - c]
    X_neg = X.iloc[new_unlabeled_samples, :]
    y_neg = y[new_unlabeled_samples]
    s_neg = np.zeros(len(new_unlabeled_samples))

    X_new = pd.concat([X_pos, X_neg])
    y_new = np.concatenate([y_pos, y_neg])
    s = np.concatenate([s_pos, s_neg])

    n_s1 = np.sum(s == 1)
    n_y1 = np.sum(y_new == 1)

    real_c = n_s1 / n_y1
    return X_new, y_new, s, real_c
