import os
import numpy as np
from matplotlib import pyplot 
from scipy import optimize
from scipy.io import loadmat
import utils

data = loadmat(os.path.join('Data', 'ex3data1.mat'))
X, y = data['X'], data['y'].ravel()
y[y==10] = 0
m = y.size

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10

weights = loadmat(os.path.join('Data', 'ex4weights.mat'))
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

Theta2 = np.roll(Theta2, 1, axis=0)
nn_params = np.concatenate([Theta1.ravel(), Theta2.ravel()])


def nnCostFunction(nn_params,
                    input_layer_size,
                    hidden_layer_size,
                    num_labels,
                    X, y, lambda_=0.0):

    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, (input_layer_size + 1)))
    Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],
                        (num_labels, (hidden_layer_size + 1)))

    m = y.size

    J = 0 
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    temp1 = Theta1
    temp2 = Theta2
    
    a1 = np.concatenate([np.ones((m,1)), X], axis=1)
    a2 = utils.sigmoid(np.dot(a1, Theta1.T))
    a2 = np.concatenate([np.ones((m,1)), a2], axis=1)
    a3 = utils.sigmoid(np.dot(a2, Theta2.T))
    
    y_matrix = y.reshape(-1)
    y_matrix = np.eye(num_labels)[y_matrix]

    reg = (lambda_/(2*m)) * (np.sum(np.square(temp1[:, 1:])) + np.sum(np.square(temp2[:, 1:])))
    J = (-1 / m) * np.sum((y_matrix * np.log(a3)) + (1 - y_matrix) * np.log(1 - a3)) + reg

    delta_3 = a3 - y_matrix
    delta_2 = np.dot(delta_3, Theta2)[:, 1:] * sigmoidGradient(np.dot(a1, Theta1.T))

    Delta1 = np.dot(delta_2.T, a1)
    Delta2 = np.dot(delta_3.T, a2)
    
    Theta1_grad = (1/m) * Delta1
    Theta1_grad[:, 1:] = Theta1_grad[:, 1:] + (lambda_/m) * Theta1[:, 1:]
    
    Theta2_grad = (1/m) * Delta2
    Theta2_grad[:, 1:] = Theta2_grad[:, 1:] + (lambda_/m) * Theta2[:, 1:]
    
    grad = np.concatenate([Theta1_grad.ravel(), Theta2_grad.ravel()])

    return J, Grad


def sigmoidGradient(z):
    g = np.zeros(z.shape)
    g = utils.sigmoid(z) * (1-utils.sigmoid(z))
    return g

def random_ini_weights(L_in, L_out, epsilon_init=0.12):
    W = np.zeros((L_out, 1+ L_in))
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W

if __name__ == '__main__':
    pass
    