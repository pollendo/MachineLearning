"""
This short program fits a bayesian linear (polynomial) regression for the sine function.
It is linear in terms of the weights, polynomial in terms of the feature matrix Phi.
In addition we plot the predictive distribution's mean and show the 1-sigma variance.
"""

import matplotlib.pyplot as plt
import numpy as np
import math
import pylab
import polynomial_regression


def gen_sine2(n):
    x = np.random.uniform(0, 2 * math.pi, n)
    t = np.zeros(n)
    x = np.asarray(sorted(x))

    for i in range(0, n):
        t[i] = np.random.normal(np.sin(x[i]), 0.25)
    return x, t


def fit_polynomial_bayes(x, t, M, alpha, beta):
    Phi = polynomial_regression.design_matrix(x, M)
    covariance_S = np.linalg.inv(alpha*np.identity(M+1)+beta*Phi.T.dot(Phi))
    m = beta * covariance_S.dot(Phi.T).dot(t)
    return m, covariance_S, Phi


def predict_polynomial_bayes(x, m, S, beta):
    Phi = polynomial_regression.design_matrix(x, 4)
    mean = np.zeros(len(x))
    sigma = np.zeros(len(x))
    n = 0
    for i in Phi:
        Phi_x = i
        mean[n] = m.T.dot(Phi_x)
        sigma[n] = ((1/beta) + (Phi_x.T).dot(S).dot(Phi_x))
        n += 1
    return mean, sigma, Phi


def plot_predictive_distribution():
    x_values, y_values = gen_sine2(10)
    order_m = 4
    alpha = 0.4
    beta = 16

    plt.figure()
    plt.title("Plot of predictive distribution for M=4, alpha=0.4 and beta=16")
    plt.xlabel("x")
    plt.ylabel("t")

    sin_wave = np.linspace(0, 2 * math.pi, 1000)
    plt.plot(sin_wave, np.sin(sin_wave), color='g', label="sine")

    mean, covariance_S, design_matrix_Phi = fit_polynomial_bayes(x_values, y_values, order_m, alpha, beta)
    pred_mean, pred_S, pred_Phi = predict_polynomial_bayes(sin_wave, mean, covariance_S, beta)

    plt.plot(x_values, y_values, linestyle="", marker='o', color='b', mfc='none', label="data")
    plt.plot(sin_wave, pred_mean, color='r', label="fit")
    plt.fill_between(sin_wave, pred_mean - pred_S, pred_mean + pred_S, color='blue', alpha=0.1, label="1-sigma-variance")
    pylab.legend(loc='lower left')
    plt.show()


plot_predictive_distribution()
