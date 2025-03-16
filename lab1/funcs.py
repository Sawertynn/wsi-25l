# WSI LAB 1
# Tomasz Kurzela

import numpy as np


def f(x, x2 = None):
    if x2 is not None:
        return x ** 2 + x2 ** 2
    x = np.asarray(x)
    return np.sum(x**2, axis=-1)


def f_grad(X):
    X = np.asarray(X)
    x1, x2 = X[..., 0], X[..., 1]
    return np.stack((2 * x1, 2 * x2), axis=-1)


def Matyas(x, x2 = None):
    if x2 is not None:
        return 0.26 * (x**2 + x2**2) - 0.48 * x * x2
    x = np.asarray(x)
    return 0.26 * np.sum(x**2, axis=-1) - 0.48 * np.prod(x, axis=-1)


def Matyas_grad(X):
    X = np.asarray(X)
    x1, x2 = X[..., 0], X[..., 1]
    grad_x1 = 0.52 * x1 - 0.48 * x2
    grad_x2 = 0.52 * x2 - 0.48 * x1
    return np.stack((grad_x1, grad_x2), axis=-1)


def f2(x1, x2):
    return x1**2 + x2**2


def f_grad2(x1, x2):
    return (2 * x1, 2 * x2)


def Matyas2(x1, x2):
    return 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2


def Matyas_grad2(x1, x2):
    return (0.52 * x1 - 0.48 * x2, 0.52 * x2 - 0.48 * x1)
