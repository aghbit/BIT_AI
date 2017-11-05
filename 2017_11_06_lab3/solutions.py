import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _hypotheses(W, X):
    result = X.dot(W)
    return sigmoid(result)


def cost(W, X, Y, reg_param=0.1, eps=0.01):
    hypotheses = _hypotheses(W, X)
    result = Y * np.log(hypotheses  + eps) + (1 - Y) * np.log(1 - hypotheses + eps)
    result = result.mean()
    result *= -1
    # print(result)
    return result  # + reg_param * np.sum(W ** 2)


def gradient_step(W, X, Y, reg_param=0.1, learning_rate=0.01):
    H = _hypotheses(W, X)

    errors = H - Y
    # print(errors)
    epsilons = (X.T.dot(errors)) / len(errors)
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

