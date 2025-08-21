import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(z))


def reLU(z):
    np.maximum(0, z)


def step(z, threshold=0):
    return 1 if z >= threshold else 0

