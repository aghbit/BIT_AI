import numpy as np
from typing import Tuple, List

def my_loss(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray) -> float:
    total_error = 0.0
    N = len(X)
    for curr_x, curr_y in zip(X, Y):
        y_pred = w_0 + w_1 * curr_x
        error = y_pred - curr_y
        total_error += error ** 2
    loss = total_error / N
    return loss

def my_loss_vectorized(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray) -> float:
    Y_pred = w_0 + w_1 * X
    errors = (Y_pred - Y) ** 2
    loss = errors.mean()
    return loss    

def dLdw_0(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray) -> float:
    Y_pred = w_0 + w_1 * X
    errors = Y_pred - Y
    return errors.mean() * 2

def dLdw_1(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray) -> float:
    Y_pred = w_0 + w_1 * X
    errors = Y_pred - Y
    errors_mul = errors * X
    return errors_mul.mean() * 2

def gradient_step_naive(
    w_0: float, 
    w_1: float, 
    X: np.ndarray, 
    Y: np.ndarray, 
    learning_rate: float
) -> Tuple[float, float, float]:
    loss = my_loss(w_0, w_1, X, Y)
    N = len(X)
    
    delta_w_1 = 0.0
    delta_w_0 = 0.0
    
    for curr_x, curr_y in zip(X, Y):
        y_pred = w_1 * curr_x + w_0 
        error =  y_pred - curr_y
        delta_w_1 += error * curr_x
        delta_w_0 += error
        
    delta_w_1 *= (2/N)
    delta_w_0 *= (2/N)
    w_1 = w_1 - learning_rate * delta_w_1
    w_0 = w_0 - learning_rate * delta_w_0
    return w_0, w_1, loss

def gradient_step_vectorized(
    w_0: float, 
    w_1: float, 
    X: np.ndarray, 
    Y: np.ndarray, 
    learning_rate: float
) -> Tuple[float, float, float]:
    loss = my_loss_vectorized(w_0, w_1, X, Y)
    delta_w_0 = dLdw_0(w_0, w_1, X, Y) * learning_rate
    delta_w_1 = dLdw_1(w_0, w_1, X, Y) * learning_rate
    w_0 = w_0 - delta_w_0
    w_1 = w_1 - delta_w_1
    return w_0, w_1, loss

def train_model(
    init_w_0: float,
    init_w_1: float,
    X: np.ndarray,
    Y: np.ndarray,
    learning_rate: float,
    num_iterations: int
) -> Tuple[float, float, List[float]]:
    w_0 = init_w_0
    w_1 = init_w_1
    loss_history = [] 
    for i in range(num_iterations):
        w_0, w_1, loss = gradient_step_vectorized(w_0, w_1, X, Y, learning_rate)
        loss_history.append(loss)
    return w_0, w_1, loss_history

X_hard = np.linspace(1, 10)
noise = np.random.normal(size=X_hard.shape)
Y_hard = 10 * np.sin(X_hard) + 5 * X_hard - 10 + 3 * noise

