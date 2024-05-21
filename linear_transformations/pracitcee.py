import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


e1 = np.array([[1], [0]])
e2 = np.array([[0], [1]])


def T(v):
    w = np.zeros((3,1))
    w[0,0] = 3*v[0,0]
    w[2,0] = -2*v[1,0]

def T_hscaling(v):
    A = np.array([2,0], [0,1])
    w = A @ v
    return w

def transform_vectors(T, v1, v2):
    V = np.hstack((v1, v2))
    W = T(V)

    return W

def T_reflection_yaxis(v):
    A = np.array([[-1, 1], [0, 1]])
    w = A @ v
    return w 

def T_stretch(a,v):
    T = np.array([[a*1,0], [0,a*1]])
    w = T @ v

    return w

def T_hshear(m, v):

    T = np.array([[1, 0], [m, 1]])
    w = T @ v
    return w

