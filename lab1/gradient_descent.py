# WSI LAB 1
# Tomasz Kurzela


import numpy as np

def gradient_descent(grad_func, starting_point, beta, max_iter=1_000, eps=1e-5, minimize=True, mark_trajectory=False):
    point = starting_point
    dir = -1 if minimize else 1
    trajectory = [point]
    step = beta * grad_func(starting_point)

    for iter in range(max_iter):
        step = dir * beta * grad_func(point)
        point = point + step
        trajectory.append(point)
        if np.linalg.norm(step) < eps:
            break

    if mark_trajectory:
        return np.array(trajectory)
    return point
