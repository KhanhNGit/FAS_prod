import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    exp_x = exp_x / exp_x.sum(axis=0)
    exp_x = np.round(exp_x*100, 2)
    return exp_x