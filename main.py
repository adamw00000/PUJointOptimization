# %%
import datasets
from functions import create_s, joint_risk, predict_proba

c = 0.8
X, y = datasets.load_spambase()
s = create_s(y, c)

X, y, s

# %%
from functions import preprocess

X_train, X_test, y_train, y_test, s_train, s_test = preprocess(X, y, s, test_size = 0.2)

# %%
import numpy as np
from functions import oracle_method, predict_proba, accuracy, oracle_risk

b = oracle_method(X_train, y_train)

y_proba = predict_proba(X_test, b)
y_pred = np.where(y_proba > 0.5, 1, 0)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

y_proba_oracle = y_proba

# %%
import numpy as np
from functions import oracle_method, predict_proba, accuracy, oracle_risk

b = oracle_method(X_train, s_train)

s_proba = predict_proba(X_test, b)
y_proba = s_proba / c
y_pred = np.where(y_proba > 0.5, 1, 0)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)

# %%
import numpy as np
from functions import joint_method, predict_proba, accuracy, oracle_risk

params = joint_method(X_train, s_train)

b = params[:-1]
c_estimate = params[-1]

y_proba = predict_proba(X_test, b)
y_pred = np.where(y_proba > 0.5, 1, 0)

risk = joint_risk(params, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))
print('Estimated c:', c_estimate)

estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)

# %%
import numpy as np
from functions import cccp_method, predict_proba, accuracy, oracle_risk

params = cccp_method(X_train, s_train)

b = params[:-1]
c_estimate = params[-1]

y_proba = predict_proba(X_test, b)
y_pred = np.where(y_proba > 0.5, 1, 0)

risk = joint_risk(params, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))
print('Estimated c:', c_estimate)

estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)












# %%
import numpy as np
from sklearn.linear_model import LogisticRegression

estimator = LogisticRegression(penalty = 'none')
estimator = estimator.fit(X_train, y_train)

b = np.insert(estimator.coef_, 0, estimator.intercept_)

y_proba = predict_proba(X_test, b)
y_pred = np.where(y_proba > 0.5, 1, 0)

risk = oracle_risk(b, X_test, y_test)
print('Risk value:', risk)
print('Accuracy:', accuracy(y_pred, y_test))

# %%
import numpy as np
from functions import create_s

s = create_s(y, 0.8)
np.sum(s == 1) / np.sum(y == 1)

# %%




# %%
import cvxpy as cvx
import numpy as np
import dccp

b = cvx.Variable(X_train.shape[1] + 1)
t = cvx.Variable(1)

expression = 0
t_expression = 0

X_mod = np.insert(X_train, 0, 1, axis = 1)
n = X_mod.shape[0]

expression += s_train * cvx.log(c)
e1 = s_train * cvx.log(c)

expression += cvx.multiply(s_train, X_mod @ b - cvx.logistic(X_mod @ b))
e2 = cvx.multiply(s_train, X_mod @ b - cvx.logistic(X_mod @ b))

expression += cvx.multiply(s_train - 1, cvx.logistic(X_mod @ b))
e3 = cvx.multiply(s_train - 1, cvx.logistic(X_mod @ b))

t_expression += cvx.multiply(s_train - 1, cvx.logistic(cvx.log(1 - c) + X_mod @ b))
e4 = cvx.multiply(1 - s_train, cvx.logistic(cvx.log(1 - c) + X_mod @ b))

print(e1.curvature, e2.curvature, e3.curvature, e4.curvature)

# for i in range(len(s_train)):
#     expression += s[i] * cvx.log(c)
#     e1 = s[i] * cvx.log(c)

#     xi = X_mod[i, :].reshape(1, -1)
#     expression += s[i] * (xi @ b - cvx.logistic(xi @ b))
#     e2 = s[i] * (xi @ b - cvx.logistic(xi @ b))

#     expression += (s[i] - 1) * (cvx.logistic(xi @ b))
#     e3 = (s[i] - 1) * (cvx.logistic(xi @ b))

#     t_expression += -(1 - s[i]) * cvx.logistic(cvx.log(1 - c) + xi @ b)
#     e4 = -(1 - s[i]) * cvx.logistic(cvx.log(1 - c) + xi @ b)
#     print(e1.curvature, e2.curvature, e3.curvature, e4.curvature)

expression = cvx.sum(expression)
expression *= -1 / len(s_train)

t_expression = cvx.sum(t_expression)
t_expression *= 1 / len(s_train)
expression += -t

problem = cvx.Problem(cvx.Minimize(expression), [t == t_expression])

b.value = np.array(np.random.random(X_train.shape[1] + 1) / 100)
t.value = np.array([-1])

print(problem.is_dcp(), dccp.is_dccp(problem))
result = problem.solve(method = 'dccp', 
    tau = 0.05, verbose = True, 
    max_iter = 10, max_iters = 5000, ccp_times = 1, 
    solver = cvx.ECOS)

print("problem status =", problem.status)
print("b =", b.value)
print("t =", t.value)
print("cost value =", result[0])

# %%
y_proba = predict_proba(X_test, b.value)
y_pred = np.where(y_proba > 0.5, 1, 0)
print('Accuracy:', accuracy(y_pred, y_test))
estimation_error = np.mean(np.abs(y_proba - y_proba_oracle))
print('Estimation error:', estimation_error)

# %%
risk = joint_risk(np.append(np.array(b.value), c), X_test, y_test)
risk

# %%
