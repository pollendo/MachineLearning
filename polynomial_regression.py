"""
This short program does a simple linear (polynomial) regression for the sine function.
It is linear in terms of the weights, polynomial in terms of the feature matrix Phi.
It highlights how small orders of polynomials lead to underfitting while too large orders overfit.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import pylab


def gen_sine(n):
    x = np.linspace(0, 2 * math.pi, n)
    t = np.zeros(n)
    for i in range(0, n):
        t[i] = np.random.normal(np.sin(x[i]), 0.25)

    return x, t


def design_matrix(x, M):  # it is highly recommended to write a helper function that computes Phi
    x0 = np.ones([len(x), 1])
    matrix = x0
    for i in range(0, M):
        power = i+1
        xi = pow(x, power)
        matrix = np.column_stack((matrix, xi))
    return matrix


def fit_polynomial(x, t, M):
    Phi = design_matrix(x, M)
    w_ml = np.linalg.inv(Phi.T.dot(Phi)).dot(Phi.T).dot(t)
    return w_ml, Phi


def plot_fitted_polynomials(x, t, M):
    sin_wave = np.linspace(0, 2 * math.pi, 1000)
    weights, Phi = fit_polynomial(x, t, M)
    test_Phi = design_matrix(sin_wave, M)
    fit = test_Phi.dot(weights.T)
    # fig = plt.figure()
    plt.title("Fitted polynomials for M =%i" % M)
    plt.xlabel("x")
    plt.ylabel("t")

    plt.plot(sin_wave, np.sin(sin_wave), color='g', label="sine")
    plt.plot(sin_wave, fit, color='r', label="fit")
    plt.plot(x, t, linestyle="", marker='o', color='b', mfc='none', label="data")

    pylab.legend(loc='upper right')
    plt.show()


x1, t1 = gen_sine(10)
plot_fitted_polynomials(x1, t1, 0)
plot_fitted_polynomials(x1, t1, 2)
plot_fitted_polynomials(x1, t1, 4)
plot_fitted_polynomials(x1, t1, 8)
