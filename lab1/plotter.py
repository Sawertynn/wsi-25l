import numpy as np
from matplotlib import pyplot as plt


def generate_linspace(
    min_arg: float, max_arg: float, num_points: int, num_dims: int = 2
) -> np.ndarray:
    x0 = np.linspace(min_arg, max_arg, num_points)
    xall = [x0] * num_dims
    xmesh = np.meshgrid(*xall)
    # return points in a single vector
    return np.vstack(list(map(np.ravel, xmesh))).T
