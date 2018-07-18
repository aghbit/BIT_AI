import numpy as np

def my_loss(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray):
    total_error = 0.0
    N = len(X)
    for curr_x, curr_y in zip(X, Y):
        y_pred = w_0 + w_1 * curr_x
        error = y_pred - curr_y
        total_error += error ** 2
    loss = total_error / N
    return loss

def my_loss_vectorized(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray):
    Y_pred = w_0 + w_1 * X
    errors = (Y - Y_pred) ** 2
    loss = errors.mean()
    return loss    

def dLdw0(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray):
    Y_pred = w_0 + w_1 * X
    errors = (Y - Y_pred)
    return errors.mean()

def dLdw1(w_0: float, w_1: float, X: np.ndarray, Y: np.ndarray):
    Y_pred = w_0 + w_1 * X
    errors = (Y - Y_pred)
    errors_mul = errors * X
    return errors_mul.mean()

def gradient_step(x, y, w_0, w_1, alpha=0.01):
    # TO BE IMPLEMENTED
    N = len(x)
    
    delta_w_1 = 0.0
    delta_w_0 = 0.0
    
    for curr_x, curr_y in zip(x, y):
        y_pred = w_1 * curr_x + w_0 
        error =  y_pred - curr_y
        delta_w_1 += error * curr_x
        delta_w_0 += error
        
    delta_w_1 *= (2/N)
    delta_w_0 *= (2/N)
    w_1 = w_1 - alpha * delta_w_1
    w_0 = w_0 - alpha * delta_w_0
    
    return w_0, w_1



x = np.linspace(1, 10)
noise = np.random.normal(size=x.shape)
y = 2 * x + 1 + noise 

y_hard = 10 * np.sin(x) + 5 * x - 10 + 3 * noise

