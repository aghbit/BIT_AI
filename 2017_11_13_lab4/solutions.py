import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_bias_feature(X):
    return np.c_[np.ones(len(X)), X]


def _hypotheses(W, X):
    result = X.dot(W)
    return sigmoid(result)


def cost(W, X, Y, eps=0.01):
    hypotheses = _hypotheses(W, X)
    result = Y * np.log(hypotheses + eps) + (1 - Y) * np.log(1 - hypotheses + eps)
    result = result.mean()
    result *= -1
    return result


def gradient_step(W, X, Y, learning_rate=0.01):
    H = _hypotheses(W, X)
    errors = H - Y
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


def std_normalization(feature_matrix, means=None, ranges=None):
    means = feature_matrix.mean(axis=0) if means is None else means
    mins = feature_matrix.min(axis=0)
    maxs = feature_matrix.max(axis=0)

    stds = feature_matrix.std(axis=0) if ranges is None else ranges
    # we alter ranges and means vector so that x_0 remains unaffected
    stds[0] = 1
    means[0] = 0
    return (feature_matrix - means) / stds, means, stds


def accuracy(actual_predictions, model_predictions):
    equals = (actual_predictions == model_predictions).astype(int)
    return equals.sum() / equals.size


def precision(actual_predictions, model_predictions):
    equals = actual_predictions == model_predictions
    eq = equals.astype(int)
    tp = actual_predictions * eq
    neq = (equals == False).astype(int)
    fp = neq * model_predictions
    return tp.sum() / (tp.sum() + fp.sum())


def recall(actual_predictions, model_predictions):
    equals = actual_predictions == model_predictions
    eq = equals.astype(int)
    tp = actual_predictions * eq
    neq = (equals == False).astype(int)
    fn = neq * (model_predictions == 0).astype(int)
    return tp.sum() / (tp.sum() + fn.sum())


def f_score(actual_predictions, model_predictions):
    p = precision(actual_predictions, model_predictions)
    r = recall(actual_predictions, model_predictions)
    return (2 * p * r) / (p + r)


tpr = recall


def fpr(actual_predictions, model_predictions):
    equals = actual_predictions == model_predictions
    eq = equals.astype(int)
    tn = (actual_predictions == 0).astype(int) * eq
    neq = (equals == False).astype(int)
    fp = neq * model_predictions
    return fp.sum() / (fp.sum() + tn.sum())


def cost_reg(W, X, Y, l=0.1, eps=0.01):
    hypotheses = _hypotheses(W, X)
    result = Y * np.log(hypotheses + eps) + (1 - Y) * np.log(1 - hypotheses + eps)
    result = result.mean()
    result *= -1
    bias_mask = np.ones(W.size)
    bias_mask[0] = 0
    return result + (l / len(Y)) * np.sum((W ** 2) * bias_mask)


def gradient_step_reg(W, X, Y, learning_rate=0.01, l=0.1):
    H = _hypotheses(W, X)
    errors = H - Y
    bias_mask = np.ones(W.size)
    bias_mask[0] = 0
    epsilons = (X.T.dot(errors) + (l * W * bias_mask)) / len(errors)
    return W - epsilons * learning_rate
