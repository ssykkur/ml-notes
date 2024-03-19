import os
import numpy as np 
from matplotlib import pyplot as plt 
from scipy import optimize
from scipy.io import loadmat
from utils import sigmoid
from logisticreg2 import costfuncreg

input_layer_size = 400
num_labels = 10

data = loadmat(os.path.join('Data', 'ex3data1.mat'))

X, y = data['X'], data['y'].ravel()

y[y==10] = 0

m = y.size



def one_vs_all(X, y, num_labels, lambda_):

    m, n = X.shape
    all_theta = np.zeros((num_labels, n + 1))
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    for c in range(num_labels):
        initial_theta = np.zeros(n+1)
        options = {'maxiter':50}

        res = optimize.minimize(costfuncreg,
                                initial_theta,
                                (X, (y==c), lambda_),
                                jac=True,
                                method='CG',
                                options=options)
        
        all_theta[c] = res.x

    return all_theta


def predict_one_vs_all(all_theta, X):
    m = X.shape[0];

    num_labels = all_theta.shape[0]
    p = np.zeros(m)

    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    
    temp = sigmoid(np.dot(X, all_theta.T))
    p = np.argmax(temp, axis=1)

    return p
    

if __name__ == '__main__':

    all_theta = one_vs_all(X, y, num_labels, lambda_=0.1)
    pred = predict_one_vs_all(all_theta, X)

    print(f'Training Set Accuracy: {np.mean(pred == y) * 100}%')