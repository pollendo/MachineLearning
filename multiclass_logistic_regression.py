from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt


def fetch_data():
    mnist = fetch_mldata('MNIST original', data_home='.')
    data, target = mnist.data, mnist.target.astype('int')
    # Shuffle
    indices = np.arange(len(data))
    np.random.seed(123)
    np.random.shuffle(indices)
    data, target = data[indices].astype('float32'), target[indices]
    # Normalize the data between 0.0 and 1.0:
    data /= 255.

    return data, target


def plot_digits(data, num_cols, targets=None, shape=(28, 28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits / num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(data[i].reshape(shape), interpolation='none', cmap='Greys')
        if targets is not None:
            plt.title(int(targets[i]))
        plt.colorbar()
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# b = 10x1
# x = 1 x 784
# t = 1
# w = 784 x 10
def logreg_gradient(x, t, w, b):
    dL_dw = np.zeros(b.shape[0])
    dL_db = np.zeros(b.shape[0])
    b = b.reshape((b.shape[0]), 1)
    log_q = np.dot(w.T, x.T) + b
    a = max(log_q)
    logZ = (a + np.log(sum(np.exp(log_q - a))))
    Z = np.exp(logZ)

    logp = (log_q.T - logZ)
    it = t[0]

    dL_db[0:it] = (-np.exp(log_q[0:it]) / Z).T[0]
    dL_db[it] = (1 - (np.exp(log_q[it]) / Z)).T[0]
    dL_db[it + 1:] = (-np.exp(log_q[it + 1:]) / Z).T[0]
    dL_dw = (np.outer(x, dL_db))
    # here the statement contains logp[:,t] where logp is meant as a matrix of shape 1x10
    return logp[:, t].squeeze(), dL_dw, dL_db.squeeze()


# Performs one iteration of stochastic gradient descent
def sgd_iter(x_train_set, t_train_set, W, b):
    alpha = 1E-4
    train_index = np.arange(len(x_train_set), dtype=int)
    train_index = np.random.permutation(train_index)
    logp_train = []
    for i in train_index:
        x = x_train_set[i].reshape((1, x_train_set[i].shape[0]))
        logpt, dw, db = logreg_gradient(x, [t_train_set[i]], W, b)
        W = W + (alpha * dw)
        b = b + (alpha * db)
        logp_train.append(logpt)
    return logp_train, W, b


def cond_prob(x, t, w, b):
    b = b.reshape((b.shape[0]), 1)
    logp_c = np.zeros(t.shape[0])
    log_q = np.dot(w.T, x.T) + b

    for i in range(t.shape[0]):
        row = log_q.T[i]
        sumt = max(row)
        log_Z = sumt + np.log(np.sum(np.exp(row - sumt), axis=0))
        logp = row - log_Z
        ind = t[i]
        logp_c[i] = logp[ind]
    return logp_c


# Performs stochastic gradient descent until convergence,
# in this case until only minor changes in conditional log probability occur (differences <= 0.005)
def test_sgd(x_train_set, t_train_set, x_valid_set, t_valid_set, w, b):
    logp_train = []
    logp_valid = []
    logptvm1 = cond_prob(x_valid_set, t_valid_set, w, b)
    logptvm1 = np.mean(logptvm1)
    logptvm2 = 10000  # arbitrary large value to enter the while loop
    i = 0
    while abs(logptvm1 - logptvm2) > 0.005:
        print("Iteration: ", i + 1)
        logptvm1 = logptvm2
        logpt, w, b = sgd_iter(x_train_set, t_train_set, w, b)
        logpv = cond_prob(x_valid_set, t_valid_set, w, b)
        logp_train.append(np.mean(logpt))
        logp_valid.append(np.mean(logpv))
        logptvm2 = np.mean(logpv)
        i = i + 1
    n = i
    print("Convergence at: ", n)
    return w, b


# Plot the weights resulting from our training
def plot_digit(data, num_cols, targets=None, shape=(28, 28)):
    num_digits = data.shape[0]
    num_rows = int(num_digits/num_cols)
    for i in range(num_digits):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(data[i].reshape(shape), interpolation='none', cmap='Greys', vmin=-.3, vmax=0.3)
        if targets is not None:
            plt.title(int(targets[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()


np.random.seed(1243)
# Split the MNIST data into training set, validation set and test set
x_train, x_valid, x_test = fetch_data()[0][:50000], fetch_data()[0][50000:60000], fetch_data()[0][60000: 70000]
t_train, t_valid, t_test = fetch_data()[1][:50000], fetch_data()[1][50000:60000], fetch_data()[1][60000: 70000]
initial_b = np.zeros(10)
initial_weights = np.zeros((28 * 28, 10))
final_weights, final_b = test_sgd(x_train, t_train, x_valid, t_valid, initial_weights, initial_b)
plot_digit(final_weights.T, num_cols=5)
