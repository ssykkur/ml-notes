import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import optimize




def plot_data(X, y, xlabe, ylabe):
    fig = plt.figure()

    pos = y==1
    neg = y==0

    plt.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mew=1)
    plt.xlabel(xlabe)
    plt.ylabel(ylabe)
    plt.show()


def sigmoid(z):
    z = np.array(z)
    g = np.zeros(z.shape)

    g = 1/(1 + np.exp(-z))
    
    return g


def cost_grad(theta, X, y):
    m = y.size

    grad = np.zeros(theta.shape)
    h = sigmoid(np.dot(X, theta))
    J = 0

    J = -(1/m) * (np.dot(y, np.log(h)) + np.dot((1 - y), np.log(1-h)))
    grad = (1/m) * (h - y).dot(X)

    return J, grad


def predict(X, theta):
    m = X.shape[0]
    p = np.zeros(m)

    p = np.round(sigmoid(np.dot(X, theta.T)))
    return p



if __name__ == '__main__':

    data = np.loadtxt(os.path.join('Data', 'ex2data1.txt'), delimiter=',')
    X, y = data[:, 0:2], data[:, 2]
    m, n = X.shape
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    initial_theta = np.zeros(n + 1)
    options = {'maxiter': 400}

    res = optimize.minimize(cost_grad,
                        initial_theta,
                        (X, y),
                        jac=True,
                        method='TNC', 
                        options=options)

    cost = res.fun
    theta = res.x

    prob = predict(np.array([1, 45, 85]), theta)

    p = predict(theta, X)
 
    s = np.mean(p==y)
    print(s)
