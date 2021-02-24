# %%
import pandas as pd
import re

def read_names_file(filename):
    with open(filename, 'r') as f:
        columns = []
        while True:
            s = f.readline()
            if s is '':
                break

            match = re.match('([^:]+):\s+[a-zA-Z]+\.', s)
            
            if match is not None:
                column_name = match.groups()[0]
                columns.append(column_name)
            
        return columns

def load_spambase():
    dataset = pd.read_csv('data/spambase.data')
    columns = read_names_file('data/spambase.names')

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X.columns = columns
    return X, y

# %%
load_spambase()

# %%
