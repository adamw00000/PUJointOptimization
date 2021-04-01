import os
import re
from scipy.io import arff

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
        # 'Adult',
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


# def load_spambase():
#     dataset = pd.read_csv(os.path.join(dir_path, 'data', 'spambase.data'))
#     columns = read_names_file(os.path.join(dir_path, 'data', 'spambase.names'))
#
#     X = dataset.iloc[:, :-1]
#
#     X_normalized = (X - X.mean()) / X.std()
#
#     y = dataset.iloc[:, -1]
#
#     X_normalized.columns = columns
#     return X_normalized, y


def load_dataset(name):
    data = arff.loadarff(os.path.join(dir_path, 'data', f'{name}.arff'))
    df = pd.DataFrame(data[0])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X, y
