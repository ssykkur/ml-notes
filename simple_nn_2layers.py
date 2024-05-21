import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.datasets import make_blobs

#nn model without regularization for practice

#dataset
m = 2000
samples, labels = make_blobs(n_samples=m,
                            centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]),
                            cluster_std=1.1,
                            random_state=0)

labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1,m))
plt.scatter(X[0,:], X[1,:], c=Y, cmap=colors.ListedColormap(['blue', 'red']));
#plt.show()

def sigmoid(z): 
    res = 1/(1 + np.exp(-z))
    return res


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 2
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)


def initialize_parameters(n_x, n_h, n_y):

    W1 = np.random.randn(n_h, n_x) * 0.1
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters


def forward_propagation(X, parameters):

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache


def compute_cost(A2, Y):

    m = Y.shape[1]
    logloss = 1/m * (np.dot(-Y, np.log(A2).T) - np.dot((1-Y), np.log(1-A2).T))
    cost = float(logloss)
    return cost


def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    
    dZ2 = A2 - Y                                        #der wrt z2
    dW2 = 1/m * np.dot(dZ2, A1.T)                       #grad of w2
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)  #grad of b2

    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)             #der wrt z1
    dW1 = 1/m * np.dot(dZ1, X.T)                        #grad of w1
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)  #grad of b1

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(parameters, grads, learning_rate=1.2):

    W1 = parameters["W1"]
    b1 = parameters['b1']
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, parameters, num_iterations=10, learning_rate=1.2, print_cost=False):

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, parameters)
        
        cost = compute_cost(A2, Y)
       
        grads = backward_propagation(parameters, cache, X, Y)
      
        parameters = update_parameters(parameters, grads)

        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters


def predict(X, parameters):

    A2, cache = forward_propagation(X, parameters)
    predictions = A2 >= 0.5
   
    return predictions

def plot_decision_boundary(predict, parameters, X, Y):
    # Define bounds of the domain.
    min1, max1 = X[0, :].min()-1, X[0, :].max()+1
    min2, max2 = X[1, :].min()-1, X[1, :].max()+1
    # Define the x and y scale.
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # Create all of the lines and rows of the grid.
    xx, yy = np.meshgrid(x1grid, x2grid)
    # Flatten each grid to a vector.
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((1, len(r1))), r2.reshape((1, len(r2)))
    # Vertical stack vectors to create x1,x2 input for the model.
    grid = np.vstack((r1,r2))
    # Make predictions for the grid.
    predictions = predict(grid, parameters)
    # Reshape the predictions back into a grid.
    zz = predictions.reshape(xx.shape)
    # Plot the grid of x, y and z values as a surface.
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral.reversed())
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));





if __name__ == '__main__':
    n_x, n_h, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)
    A2, cache = forward_propagation(X, parameters)
    cost = compute_cost(A2, Y)
    grads = backward_propagation(parameters, cache, X, Y)
    parameters_trained = nn_model(X, Y, n_h, parameters) 
    
    X_pred = np.array([[2, 8, 2, 8], [2, 8, 8, 2]])
    Y_pred = predict(X_pred, parameters_trained)


