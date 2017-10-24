import numpy as np

def cost(x, y, w_0, w_1):
    # TO BE IMPLEMENTED
    w_0, w_1
    cost = 0.0
    N = len(x)
    for curr_x, curr_y in zip(x, y):
        y_pred = w_0 + w_1 * curr_x
        error = y_pred - curr_y
        cost += error ** 2
    average_cost = cost / N
    return average_cost


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
