import os
import numpy as np 
from matplotlib import pyplot as plt 
from scipy import optimize
from scipy.io import loadmat
import utils

input_layer_size = 400
num_labels = 10

data = loadmat(os.path.join('Data', 'ex3data1.mat'))

X, y = data['X'], data['y'].ravel()

y[y==10] = 0

m = y.size