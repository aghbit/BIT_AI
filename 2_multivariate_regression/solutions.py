import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

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

def train_model(
    init_W: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    learning_rate: float,
    num_iterations: int
) -> Tuple[np.ndarray, List[float]]:
    W = init_W
    loss_history = []
    for i in range(num_iterations):
        loss_history.append(loss(W, X, Y))
        W = gradient_step(W, X, Y, learning_rate)
    return W, loss_history
        

def mean_normalization(feature_matrix, means=None, ranges=None):
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)
    means = feature_matrix.mean(axis=0) if means is None else means
    ranges = maxs - mins if ranges is None else ranges
    # we alter ranges and means vector so that x_0 remains unaffected
    ranges[0] = 1
    means[0] = 0
    return (feature_matrix - means) / ranges, means, ranges

X_1 = np.linspace(1, 10) 
noise = np.random.randn(X_1.shape[0]) 
Y_1 = 2.1 * X_1 + 3.7 + noise