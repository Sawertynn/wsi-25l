# WSI LAB 1
# Tomasz Kurzela

import numpy as np
from matplotlib import pyplot as plt

from funcs import f, f_grad, Matyas, Matyas_grad
from gradient_descent import gradient_descent


## Part 1: different beta values
betas = [0.1, 0.2, 0.5, 0.7]

fig, axes = plt.subplots(1, 2)

clumped = [(f, f_grad), (Matyas, Matyas_grad)]

for ax, (fun, grad) in zip(axes, clumped):
    for beta in betas:
        trajectory = gradient_descent(
            grad, [10, 10], beta=beta, max_iter=200, eps=1e-3, mark_trajectory=True
        )
        values = fun(trajectory)
        ax.plot(values, label=f"{beta}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Evalutaion")
        ax.set_title(fun.__name__)
        ax.legend()

fig.set_size_inches(20, 10)
plt.show()

## Part 2: plotting trajectory

MIN_X = -10
MAX_X = 10
PLOT_SIZE = 501
PLOT_STEP = 0.1

BETA = 0.3

starting_points = np.array([[10, 10], [-10, 10], [0, -10]])


x1 = np.linspace(MIN_X, MAX_X, PLOT_SIZE)

X1, X2 = np.meshgrid(x1, x1)


for fun, grad in clumped:
    Z = fun(X1, X2)
    fig, axes = plt.subplots(1, len(starting_points))
    for idx, sp in enumerate(starting_points):
        trajectory = gradient_descent(grad, sp, beta=BETA, mark_trajectory=True)
        pcm = axes[idx].pcolormesh(X1, X2, Z, cmap="viridis", shading="auto")

        fig.colorbar(pcm, ax=axes[idx])

        axes[idx].plot(
            trajectory[:, 0], trajectory[:, 1], marker=".", color="red", alpha=0.25
        )

        axes[idx].set_aspect(1.0)

    fig.set_size_inches(20, 6)
    fig.tight_layout()
    fig.suptitle(fun.__name__)
    plt.show()
