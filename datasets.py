import os
import re

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


def load_spambase():
    dataset = pd.read_csv(os.path.join(dir_path, 'data', 'spambase.data'))
    columns = read_names_file(os.path.join(dir_path, 'data', 'spambase.names'))

    X = dataset.iloc[:, :-1]

    X_normalized = (X - X.mean()) / X.std()

    y = dataset.iloc[:, -1]

    X_normalized.columns = columns
    return X_normalized, y
