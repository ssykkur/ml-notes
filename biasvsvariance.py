import os
import numpy as np
import utils
from scipy import optimize
from matplotlib import pyplot
from scipy.io import loadmat


data = loadmat(os.path.join('Data', 'ex5data1.mat'))

X, y = data['X'], data['y'][:, 0]
Xtest, ytest = data['Xtest'], data['ytest'][:, 0]
Xval, yval = data['Xval'], data['yval'][:, 0]
pyplot.plot(X, y, 'ro', ms=10, mec='k', mew=1)
pyplot.xlabel('Change in water level (x)')
pyplot.ylabel('Water flowing out of the dam (y)')
m = y.size

def linearRegCostFunction(X, y, theta, lambda_=0.0):
    m = y.size

    J = 0
    grad = np.zeros(theta.shape)
    inner = np.power((X.dot(theta) - y),2)

    J = (np.sum(inner))/(2*m) + ((lambda_ /(2*m)) * np.sum(np.power(theta[1:],2)))
    temp = (X.dot(theta) - y) 
    grad = (np.dot(X.T, temp))/m  
    grad[1:] = grad[1:] + (lambda_ * theta[1:])/m
    return J, grad


def learningCurve(X, y, Xval, yval, lambda_=0):
    m = y.size

    error_train = np.zeros(m)
    error_val = np.zeros(m)

    for i in range(1, m+1):
        theta_t = utils.trainLinearReg(linearRegCostFunction, X[:i], y[:i], lambda_ = lambda_)
        error_train[i-1], _ = linearRegCostFunction(X[:i], y[:i], theta_t, lambda_ = 0)
        error_val[i-1], _ = linearRegCostFunction(Xval, yval, theta_t, lambda_ = 0)

    return error_train, error_val


def poly_features(X, p):

    X_poly = np.zeros((X.shape[0], p))

    for i in range(p):
        X_poly[:, i] = X[:, 0] ** (i + 1)

    return X_poly 


if __name__ == '__main__':
    #training set error test
    mtest = Xtest.shape[0]
    mtraining = X.shape[0]
    X_poly = poly_features(X, 8)
    X_poly, _, _ = utils.featureNormalize(X_poly)
    X_poly = np.concatenate([np.ones((mtraining, 1)), X_poly], axis=1)
    
    theta_t = utils.trainLinearReg(linearRegCostFunction, X_poly, y, lambda_=3)
    


    Xtest_poly = poly_features(Xtest, 8)
    Xtest_poly, _, _ = utils.featureNormalize(Xtest_poly)
    Xtest_poly = np.concatenate([np.ones((mtest, 1)), Xtest_poly], axis=1)

    Xtest_error, _ = linearRegCostFunction(Xtest_poly, ytest, theta_t, lambda_= 0)

    print(Xtest_error)