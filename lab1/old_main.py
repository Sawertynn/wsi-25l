import numpy as np
from matplotlib import pyplot as plt

from funcs import f, f_grad, Matyas, Matyas_grad, Matyas2, Matyas_grad2, f2
from gradient_descent import gradient_descent
from plotter import generate_linspace

MIN_X = -10
MAX_X = 10
PLOT_SIZE = 501
PLOT_STEP = 0.1

clumped_funcs = [
    [f, f_grad, f2],
    [Matyas, Matyas_grad, Matyas2]
]

# starting_points = [
#     np.array([5, -5])
# ]

starting_points = np.asarray([
    [8, 6],
    [5, -9],
    [-10, 1],
    [4, 2],
    [9, 9],
    [0, 0]
])

betas = [
    0.1,
    0.3,
    0.8,
    1.5
]



x1 = np.linspace(MIN_X, MAX_X, PLOT_SIZE)

X1, X2 = np.meshgrid(x1, x1)
xmesh = np.meshgrid(x1, x1)
XMESH = np.vstack(list(map(np.ravel, xmesh))).T

Y = Matyas(XMESH)

Y2 = Matyas2(X1, X2)


for b in betas:
    fig, axs = plt.subplots(2, 3)
    for sp, idx in zip(starting_points, range(6)):
        row, col = idx // 3, idx % 3
        trajectory = gradient_descent(Matyas_grad, sp, beta=b, eps=1e-4, mark_trajectory=True)
        steps = len(trajectory)
        best = trajectory[-1]

        pcm = axs[row, col].pcolormesh(X1, X2, Matyas2(X1, X2), cmap="viridis", shading="auto")
        axs[row, col].set_xlabel("x1")
        axs[row, col].set_ylabel("x2")
        axs[row, col].set_title(f"start: {sp}, iter: {steps}, best: {Matyas(best):.3f}")

        fig.colorbar(pcm, ax=axs[row, col])

        axs[row, col].plot(
            trajectory[:, 0],
            trajectory[:, 1],
            marker=".",
            color="red",
            label="Gradient Descent Steps",
            alpha=0.2,
        )

    fig.tight_layout()
    fig.suptitle(f"Matyas beta: {b}")
    fig.set_size_inches(20, 10)
    plt.savefig(f"./figures/Matyas_{b}.png")
    
