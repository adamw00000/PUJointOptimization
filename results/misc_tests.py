# %%
from datasets import get_dataset_stats

stats_df = get_dataset_stats()
stats_df.to_csv('dataset_stats.csv', index=None)

# %%
import cvxpy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

from results.test_dicts import marker_styles
from datasets import gen_probit_dataset, get_datasets
from data_preprocessing import preprocess, create_s
from optimization import JointClassifier, CccpClassifier, NaiveClassifier, WeightedClassifier, OracleClassifier, \
    MMClassifier, DccpClassifier
from optimization.c_estimation import TIcEEstimator
from optimization.functions import accuracy, joint_risk, oracle_risk, sigma, add_bias
from optimization.metrics import approximation_error, auc

n_features = 1
b = np.ones(n_features)
b_bias = np.append([0], b)

target_c = 0.5
N = 1000
X, y = gen_probit_dataset(int(N), b, n_features, include_bias=False)
# X, y = get_datasets()['vote']
s, c = create_s(y, target_c)

X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size=0,
                                                               n_best_features=n_features)

clf = DccpClassifier(tau=1, verbosity=0, tol=1e-3, max_iter=40, mosek_max_iter=1000, dccp_max_iter=1000)

clf.fit(X_train, s_train, c=c)

real_probas = sigma(np.matmul(add_bias(X_train), b_bias))
est_probas = sigma(np.matmul(add_bias(X_train), clf.params))
print(np.mean(np.abs(est_probas - real_probas)))
print(np.mean(np.where(est_probas > 0.5, 1, 0) == y_train))

dccp_params = clf.params
print(joint_risk(clf.params, add_bias(X_train), s_train, c))

pd.DataFrame({
    'RealProba': real_probas,
    'EstProba': est_probas,
    'RealClass': y_train,
    'EstClass': np.where(est_probas > 0.5, 1, 0),
}).to_csv('dccp.csv')

clf = JointClassifier(include_bias=False)
clf.fit(X_train, s_train, c=c)

real_probas = sigma(np.matmul(X_train, b))
est_probas = sigma(np.matmul(X_train, clf.params))
print(np.mean(np.abs(est_probas - real_probas)))
print(np.mean(np.where(est_probas > 0.5, 1, 0) == y_train))

pd.DataFrame({
    'RealProba': real_probas,
    'EstProba': est_probas,
    'RealClass': y_train,
    'EstClass': np.where(est_probas > 0.5, 1, 0),
}).to_csv('joint.csv')

joint_params = clf.params
print(joint_risk(clf.params, X_train, s_train, c))

clf = CccpClassifier(include_bias=False, inner_tol=1e-10)
clf.fit(X_train, s_train, c=c)

real_probas = sigma(np.matmul(X_train, b))
est_probas = sigma(np.matmul(X_train, clf.params))
print(np.mean(np.abs(est_probas - real_probas)))
print(np.mean(np.where(est_probas > 0.5, 1, 0) == y_train))

cccp_params = clf.params
print(joint_risk(cccp_params, X_train, s_train, c))

# %%
print(joint_risk(joint_params, X_train, s_train, c))
print(joint_risk(dccp_params, add_bias(X_train), s_train, c))
print(joint_risk(cccp_params, X_train, s_train, c))

# %%
from optimization.functions import cccp_risk_wrt_b

print(cccp_risk_wrt_b(joint_params, X_train, s_train, c, cccp_params))
print(cccp_risk_wrt_b(cccp_params, X_train, s_train, c, cccp_params))

# %%
from optimization.functions import cccp_risk_wrt_b

def true_oracle(b, X, y):
    n = X.shape[0]
    return -1/n * np.sum(y * np.log(sigma(X@b)) + (1 - y) * np.log(1 - sigma(X@b)))

def E_vex(b, X, s, c):
    return oracle_risk(b, X, s)

def E_cave(b, X, s, c):
    f1 = s * np.log(c)
    f3 = (1 - s) * np.logaddexp(0, np.log(1 - c) + X@b)
    # f3 = (1 - s) * np.log(1 + (1 - c) * np.exp(X@b))
    # f3 = (1 - s) * np.log(1 + (1 - c) * np.exp(X@b))
    n = X.shape[0]
    return -1/n * np.sum(f1 + f3)

v = np.linspace(-100, 100, 1000)

res_oracle = []
res_oracle_true = []
res_cccp = []
res = []
res_vex = []
res_cave = []

for el in v:
    res_oracle.append(oracle_risk([el], X_train, y_train))
    # res_oracle_true.append(true_oracle([el], X_train, y_train))
    res_cccp.append(cccp_risk_wrt_b([el], X_train, s_train, c, cccp_params))
    res.append(joint_risk([el], X_train, s_train, c))
    res_vex.append(E_vex([el], X_train, s_train, c))
    res_cave.append(E_cave([el], X_train, s_train, c))

plt.plot(v, res_oracle)
plt.title('Oracle risk')
plt.show()
# plt.plot(v, res_oracle_true)
# plt.show()

plt.plot(v, res)
plt.title('Joint risk')
plt.show()

plt.plot(v, res_vex)
plt.title('Convex component')
plt.show()

plt.plot(v, res_cave)
plt.title('Concave component')
plt.show()

plt.plot(v, np.array(res_vex) + np.array(res_cave))
plt.title('Sum of convex and concave components = joint risk')
plt.show()

plt.plot(v, res_cccp)
plt.title('Cccp function')
plt.show()

# %%
oracle_risk([100], X_train, y_train)

# %%
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

from datasets import gen_probit_dataset

sns.set_theme()
plt.figure(figsize=(8, 6))


X, y = gen_probit_dataset(100, [0, 1], 1, True)
max_x = max(np.abs(np.max(X.to_numpy())), np.abs(np.min(X.to_numpy())))

l = np.linspace(-max_x, max_x, 1000)

plt.plot(l, scipy.stats.norm.cdf(l))
plt.scatter(X.to_numpy()[:, 0], y, c='r', s=15)

plt.legend(['Model probitowy', 'Pobrane próbki'])

plt.ylabel('Prawdopodobieństwo przynależności do klasy dodatniej')
plt.title('Generowanie zbioru probitowego - przypadek jednowymiarowy')
plt.xlabel('X')
plt.savefig(r'results\standalone_plots\probit_orig.png', bbox_inches='tight', dpi=300)
plt.savefig(r'results\standalone_plots\probit_orig.svg', bbox_inches='tight')
plt.show()
plt.close()

# %%
from optimization import OracleClassifier

clf = OracleClassifier(include_bias=True)
clf.fit(X.to_numpy(), y.to_numpy())

print(clf.params)

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
from optimization.functions import accuracy, joint_risk, oracle_risk, sigma, add_bias

sns.set_theme()
plt.figure(figsize=(8, 6))

plt.plot(l, scipy.stats.norm.cdf(l), 'b--')
plt.plot(l, sigma(l), 'g-.')
plt.plot(l, sigma(clf.params[1]*l + clf.params[0]), 'g')

plt.scatter(X.to_numpy()[:, 0], y, c='r', s=15)

plt.legend(['Oryginalny model probitowy',
            'Model logistyczny z oryginalnym wektorem parametrów',
            'Dopasowany model logistyczny',
            'Próbki z rozkładu probitowego'], bbox_to_anchor=(0.28, 0.08, 0.5, 0.2))

plt.ylabel('Prawdopodobieństwo przynależności do klasy dodatniej')
plt.title('Model probitowy - dopasowanie klasyfikatora logistycznego')
plt.xlabel('X')
plt.savefig(r'results\standalone_plots\probit_best_fit.png', bbox_inches='tight', dpi=300)
plt.savefig(r'results\standalone_plots\probit_best_fit.svg', bbox_inches='tight')
plt.show()
plt.close()

# %%
def joint_risk_second_derivative(params, X, s, exact_c=None):
    n = X.shape[0]

    if exact_c is None:
        b = params[:-1]
        c = params[-1]
    else:
        b = params
        c = exact_c

    sig = sigma(np.matmul(X, b))

    # dbdb = sig * (1 - sig) * ((1-s)*(1-c) / ((1 - c * sig)*(1 - c * sig)) - 1)
    dbdb = sig * (1 - sig) * (c*c*sig*sig - 2*c*sig + s) / ((1 - c * sig)*(1 - c * sig))
    dbdb = (X.T @ X) * dbdb

    hessian = dbdb

    if exact_c is None:
        dbdc = (s - 1) * sig * (1 - sig) / ((1 - c * sig)*(1 - c * sig))
        dbdc = X * dbdc.reshape(-1, 1)

        dcdc = -s/(c*c) + sig*sig * (s - 1) / ((1 - c * sig)*(1 - c * sig))

        hessian = np.insert(hessian, dbdb.shape[0], dbdc, axis=1)
        last_row = np.concatenate([dbdc, [dcdc]])
        hessian = np.insert(hessian, dbdb.shape[0], last_row, axis=0)

    hessian *= -1/n
    return hessian

v = np.linspace(-100, 100, 100)

res_derivs = []

for el in v:
    res_derivs.append(joint_risk_second_derivative([el], X_train, s_train, c)[0][0])

plt.plot(v, res_derivs)
plt.show()

# %%
joint_risk_second_derivative([el], X_train, s_train, c)

# %%
np.sum(np.array(res_derivs) > 0)
