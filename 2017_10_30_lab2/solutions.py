import numpy as np
import time


def hypotheses(W, X):
    result = X.dot(W)
    return result


def cost(W, X, Y, reg_param=0.1):
    hypotheses = X.dot(W)
    errors = hypotheses - Y
    errors_squared = errors ** 2
    return np.mean(errors_squared) / 2  # + reg_param * np.sum(W ** 2)


def gradient_step(W, X, Y, reg_param=0.1, learning_rate=0.01):
    H = hypotheses(W, X)
    errors = H - Y
    epsilons = np.mean(np.transpose(np.multiply(errors, np.transpose(X))), axis=0)
    return W - epsilons * learning_rate


def mean_normalization(feature_matrix):
    means = feature_matrix.mean(axis=0)
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)
    ranges = maxs - mins
    # we alter ranges and means vector so that x_0 remains unaffected
    ranges[0] = 1
    means[0] = 0
    return (feature_matrix - means) / ranges


def secret_polynomial(x):
    return 0.5 * x ** 3 - 4*x + 6 * np.random.rand()

