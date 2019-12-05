import numpy as np

def preprocess(X, y):
    
    X_train, X_test, y_train, y_test = X[:10000], X[10000:], y[:10000], y[10000:]

    shuffle_index = np.random.permutation(10000)
    X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
    
    X_train, X_test = X_train/255, X_test/255
    
    return X_train.T, y_train.reshape(1, 10000), X_test, y_test

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_backward(dA, Z):
    S = sigmoid(Z)
    dS = S * (1 - S)
    return dA * dS

def relu_backward(dA, Z):
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    return dZ

def loss(dY, Y, eps=1e-6):
    result = Y * np.log(dY + eps) + (1 - Y) * np.log(1 - dY + eps)
    result = result.mean()
    result *= -1
    return result


def cost(predict, actual):
    m = actual.shape[1]
    cost__ = -np.sum(np.multiply(np.log(predict), actual) + np.multiply((1 - actual), np.log(1 - predict)))/m
    return np.squeeze(cost__)

def set_params(X, Y):
    input_size = X.shape[0] 
    output_size = Y.shape[0]
    
    W1 = np.random.randn(128, input_size)*np.sqrt(2/input_size)
    b1 = np.zeros((128, 1))
        
    W2 = np.random.randn(output_size, 128)*np.sqrt(2/128)
    b2 = np.zeros((output_size, 1))

    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}

def forward_propagation(X, params):
    
    Z1 = params['W1'] @ X + params['b1']
    A1 = relu(Z1)
    
    Z2 = params['W2'] @ A1 + params['b2']
    y = sigmoid(Z2)
    
    return y, {'Z1': Z1, 'Z2': Z2, 'A1': A1, 'y': y}

def backward_propagation(X, Y, params, cache):
    m = X.shape[1]
    
    dy = cache['y'] - Y
    
    dW2 = (1 / m) * (dy @ cache['A1'].T)
    db2 = (1 / m) * np.sum(dy, axis=1, keepdims=True)
    
    dZ1 = relu_backward(dy, cache['Z2'])
    
    dW1 = (1 / m) * (dZ1 @ X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(gradients, params, learning_rate = 1.2):
    W1 = params['W1'] - learning_rate * gradients['dW1']
    b1 = params['b1'] - learning_rate * gradients['db1']
    W2 = params['W2'] - learning_rate * gradients['dW2']
    b2 = params['b2'] - learning_rate * gradients['db2']
    return {'W1': W1, 'W2': W2, 'b1': b1, 'b2': b2}


def fit(X, Y, learning_rate, epochs = 100):
    params = set_params(X, Y)
    cost_ = []
    for _ in range(epochs):
        y, cache = forward_propagation(X, params)
        costit = cost(y, Y)
        gradients = backward_propagation(X, Y, params, cache)
        params = update_parameters(gradients, params, learning_rate)
        cost_.append(costit)
    return params, cost_