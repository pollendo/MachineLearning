"""
We are going to use Support Vector Machines to obtain the decision boundary for which the margin is maximized.
This separates our data into two clusters.
"""

import cvxopt
import matplotlib.pyplot as plt
import numpy as np


# First we create some toy data and plot each label in a different color
def create_x_and_t():
    X1 = np.random.multivariate_normal((1, 1), 0.2 * np.identity(2), 20)
    X2 = np.random.multivariate_normal((3, 3), 0.2 * np.identity(2), 30)
    plt.plot(X1[:, 0], X1[:, 1], 'bo')
    plt.plot(X2[:, 0], X2[:, 1], 'go')

    X = np.concatenate((X1, X2))
    t = np.ones(len(X))
    for i in range(len(X1)):
        t[i] = -1
    return X, t


# We compute the kernel matrix for the optimization problem
def compute_k(X):
    K = np.zeros([len(X), len(X)])
    for n in range(len(X)):
        for m in range(len(X)):
            K[n][m] = X[n].dot(X[m])
    return K


# Solves the quadratic programming problem using the cvxopt module and
# returns the lagrangian multiplier for every sample in the dataset.
def compute_multipliers(X, t):
    q = cvxopt.matrix(-1*np.ones(len(X)))
    G = cvxopt.matrix(np.identity(len(X))*-1)
    h = cvxopt.matrix(np.zeros(len(X)).T)
    b = cvxopt.matrix(np.matrix(0.0))
    t = t.reshape((len(t), 1))
    A = cvxopt.matrix(t.T)
    P = cvxopt.matrix(compute_k(np.multiply(X, t)))
    sol = cvxopt.solvers.qp(P, q, G, h, A, b)
    a = np.array(sol['x'])
    return a


def plot_support_vectors_and_decision_boundary():
    X, t = create_x_and_t()
    a = compute_multipliers(X, t)

    c = 0
    indices = []
    for i in a:
        if i > 0.4:
            indices.append(c)
            plt.scatter(X[c][0], X[c][1], marker="*", color="r", zorder=3)
        c += 1

    a = a.squeeze()
    w = np.multiply(a.T, t) @ X
    at = np.multiply(a[indices], t[indices])
    b = 1 / len(indices) * np.sum(t[indices] - at.T @ compute_k(X[indices]))

    x_line = np.linspace(np.min(X.T[0]), np.max(X.T[0]), 10)
    y_line = - w[0] / w[1] * x_line - b / w[1]

    y_line_top = - w[0] / w[1] * x_line - (b-1) / w[1]
    y_line_bot = - w[0] / w[1] * x_line - (b+1) / w[1]

    plt.plot(x_line, y_line, 'r-')
    plt.plot(x_line, y_line_top, 'r--')
    plt.plot(x_line, y_line_bot, 'r--')
    plt.show()


plot_support_vectors_and_decision_boundary()
