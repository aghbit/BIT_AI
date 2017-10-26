import numpy as np
import time


def hypotheses(W, X):
    result = X.dot(W)
    return result


def cost(W, X, Y):
    N = len(X)
    hypotheses = X.dot(W)
    errors = hypotheses - Y
    errors_squared = errors ** 2
    return np.sum(errors_squared) / (2 * N)


def gradient_step(W, X, Y, learning_rate=0.1):
    H = hypotheses(W, X)
    errors = H - Y
    epsilons = np.mean(np.transpose(np.multiply(errors, np.transpose(X))), axis=0)
    return W - epsilons * learning_rate

    # return W - learning_rate * epsilons


def slow_hypothesis(W, X):
    result = 0
    for w, x in zip(W, X):
        result += w * x
    return result


def time_of(fun, *params):
    s = time.time()
    fun(*params)
    return time.time() - s

