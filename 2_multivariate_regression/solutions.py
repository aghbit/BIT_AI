import numpy as np
import matplotlib.pyplot as plt

def hypotheses(W, X):
    return X @ W

def loss(W, X, Y):
    h = hypotheses(W, X)
    errors = h - Y
    errors_squared = errors ** 2
    return np.mean(errors_squared) / 2


def gradient_step(W, X, Y, learning_rate=0.01):
    H = hypotheses(W, X)
    errors = H - Y
    epsilons = X.transpose() @ errors / len(errors)
    return W - epsilons * learning_rate


def mean_normalization(feature_matrix, means=None, ranges=None):
    means = feature_matrix.mean(axis=0) if means is None else means
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)
    ranges = maxs - mins if ranges is None else ranges
    # we alter ranges and means vector so that x_0 remains unaffected
    ranges[0] = 1
    means[0] = 0
    return (feature_matrix - means) / ranges, means, ranges


def to_poly_features(X, proposed_degree):
    # notice that x ** 0 = 1, so bias feature is already added here
    return np.array([[x ** n for n in range(proposed_degree)] for x in X])


def secret_polynomial(X):
    return 0.5 * X ** 3 - 4 * X + 6 * np.random.rand()


def perform_polynomial_regression(steps=100, degree=4):
    secret = secret_polynomial
    X = np.arange(-4, 4, 0.7)
    Y = [secret(x) for x in X]

    features = to_poly_features(X, degree)
    targets = np.array(Y)
    features, means, ranges = mean_normalization(features)

    W = np.random.rand(degree)
    costs = []

    for i in range(steps):
        W = gradient_step(W, features, targets, 0.1)
        costs.append(loss(W, features, targets))

    step_nums = [i for i in range(steps)]
    plt.scatter(x=step_nums, y=costs)
    plt.show()

    calculated_targets = hypotheses(W, features)
    plt.scatter(X, targets)
    plt.plot(X, calculated_targets, color='red')
    plt.show()

    more_X = np.arange(-6, 6, 0.3)
    more_Y = [secret(x) for x in more_X]
    more_features = to_poly_features(more_X, degree)
    more_features, means, ranges = mean_normalization(more_features, means, ranges)
    # more_targets = np.array(Y)

    more_calculated_targets = hypotheses(W, more_features)
    plt.scatter(more_X, more_Y)
    plt.plot(more_X, more_calculated_targets, color='red')
    plt.show()
    print(W)

X_1 = np.linspace(1, 10) 
noise = np.random.randn(X_1.shape[0]) 
Y_1 = 2.1 * X_1 + 3.7 + noise