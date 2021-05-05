import os
import re
from scipy.io import arff
import numpy as np

import pandas as pd

dir_path = os.path.dirname(os.path.realpath(__file__))


def read_names_file(filename):
    with open(filename, 'r') as f:
        columns = []
        while True:
            s = f.readline()
            if s == '':
                break

            match = re.match(r'([^:]+):\s+[a-zA-Z]+\.', s)
            
            if match is not None:
                column_name = match.groups()[0]
                columns.append(column_name)
            
        return columns


def get_datasets():
    names = [
        'Adult',
        'BreastCancer',
        'credit-a',
        'credit-g',
        'diabetes',
        'heart-c',
        'spambase',
        'vote',
        'wdbc',
    ]

    return {name: load_dataset(name) for name in names}


def load_dataset(name):
    data = arff.loadarff(os.path.join(dir_path, 'data', f'{name}.arff'))
    df = pd.DataFrame(data[0])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def gen_synthetic_dataset_M1(alpha, mu, N):
    pos_size = round(alpha * N)
    neg_size = N - pos_size
    positives = np.random.normal(mu, 1, pos_size)
    negatives = np.random.normal(0, 1, neg_size)

    df = pd.DataFrame({
        'X1': np.concatenate([positives, negatives]),
    })
    for i in range(2, 11):
        df[f'X{i}'] = np.random.normal(0, 1, N)
    df['y'] = np.concatenate([np.ones(pos_size), np.zeros(neg_size)])

    df = df.sample(frac=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def gen_synthetic_dataset_M2(alpha, mu, N):
    pos_size = round(alpha * N)
    easy_pos_size = round(0.75 * pos_size)
    hard_pos_size = pos_size - easy_pos_size
    neg_size = N - pos_size

    easy_positives = np.random.normal(mu, 1, easy_pos_size)
    hard_positives = np.random.normal(0, 1, hard_pos_size)
    negatives = np.random.normal(0, 1, neg_size)

    df = pd.DataFrame({
        'X1': np.concatenate([easy_positives, hard_positives, negatives]),
    })
    for i in range(2, 11):
        df[f'X{i}'] = np.random.normal(0, 1, N)
    df['y'] = np.concatenate([np.ones(pos_size), np.zeros(neg_size)])

    df = df.sample(frac=1)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y


def gen_M1_dataset(alpha=0.5, mu=1, N=5000):
    X, y = gen_synthetic_dataset_M1(alpha, mu, N)
    return X, y

def gen_M2_dataset(alpha=0.5, mu=1, N=5000):
    X, y = gen_synthetic_dataset_M2(alpha, mu, N)
    return X, y
