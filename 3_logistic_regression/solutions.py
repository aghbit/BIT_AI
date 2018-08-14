import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def _hypotheses(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    result = X @ W
    return sigmoid(result)

def loss(
    W: np.ndarray, 
    X: np.ndarray, 
    Y: np.ndarray, 
    eps: float = 0.01
) -> float:
    hypotheses = _hypotheses(W, X)
    result = Y * np.log(hypotheses + eps) + (1 - Y) * np.log(1 - hypotheses + eps)
    result = result.mean()
    result *= -1
    return result


def gradient_step(
    W, 
    X, 
    Y,
    learning_rate=0.01
) -> np.ndarray:
    H = _hypotheses(W, X)
    errors = H - Y
    epsilons = (X.T.dot(errors)) / len(errors)
    return W - epsilons * learning_rate


def mean_normalization(
    feature_matrix: np.ndarray
) -> np.ndarray:
    means = feature_matrix.mean(axis=0)
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)
    ranges = maxs - mins
    # we alter ranges and means vector so that x_0 remains unaffected
    ranges[0] = 1
    means[0] = 0
    return (feature_matrix - means) / ranges


def std_normalization(
    feature_matrix: np.ndarray, 
    means: np.ndarray = None, 
    ranges: np.ndarray = None
) -> np.ndarray:
    means = feature_matrix.mean(axis=0) if means is None else means
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)
    
    stds = feature_matrix.std(axis=0) if ranges is None else ranges
    # we alter ranges and means vector so that x_0 remains unaffected
    stds[0] = 1
    means[0] = 0
    return (feature_matrix - means) / stds, means, stds
