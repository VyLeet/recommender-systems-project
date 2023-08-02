import numpy as np


"""
"""


def mae(a, b):
    a, b = np.array(a), np.array(b)
    return np.abs(a - b) / len(a)

def rmse(a, b):
    a, b = np.array(a), np.array(b)
    return np.sqrt((a - b)**2 / len(a))

def mapk(a, b, k):
    pass
