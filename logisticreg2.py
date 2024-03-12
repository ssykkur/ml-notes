from matplotlib import pyplot as plt
from scipy import optimize
from logisticreg import plot_data, sigmoid
import numpy as np 
import os


def mapfeature(X1, X2, degree=6):

    if X.ndim > 0:
        out = [np.ones(X1.shape[0])]
        
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))
    
    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.array(out)


def costfuncreg(theta, X, y, lambda_):
    m = y.size
    J = 0
    grad = np.zeros(theta.shape)
    
    h = sigmoid(np.dot(X, theta))

    temp = theta 
    temp[0] = 0

    J = -(1/m) * ((y.dot(np.log(h))) + (1 - y).dot(np.log(1-h))) + (lambda_/(2*m) * np.sum(np.square(temp)))

    grad = (1/m) * (h-y).dot(X)
    grad = grad + (lambda_ * temp)/m

    return J, grad



if __name__ == '__main__':
    data = np.loadtxt(os.path.join('Data', 'ex2data2.txt'), delimiter=',')
    X, y = data[:, :2], data[:, 2]
    
    plot_data(X,y,'microchip test 1','microfhip test 2')

    X = mapfeature(X[:, 0], X[:, 1])
    initial_theta = np.zeros(X.shape[1])
    test_theta = np.ones(X.shape[1])
    cost, grad = costfuncreg(initial_theta, X, y, 1)
    print(cost, grad[:5])
