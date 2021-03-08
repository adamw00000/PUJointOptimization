import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(X, y, s, test_size = 0.2):
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size = test_size)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, s_train, s_test


def create_s(y, c): # c - label_frequency
    s = np.array(y)
    positives = np.where(s == 1)[0]
    
    unlabelled_samples = positives[np.random.random(len(positives)) < 1 - c]
    s[unlabelled_samples] = 0
    return s
