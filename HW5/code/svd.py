import numpy as np
import matplotlib.pyplot as plt


def plot_n_modes(X, V, n, shape, path_out):
    """Shows the numbers represented with the selected number of SVD modes

    Args:
        X (array_like): Data matrix with rows of images
        V (array_like): Matrix with mode vectors as columns
        n (int): Number of modes to use in the representation
        shape (tuple): The shape of the image (pixels_y, pixels_x)
        path_out (str): Path to save figure to
    """
    fig, axs = plt.subplots(2, figsize=(3, 3))

    selected = 10
    Y = np.dot(V, X[selected].T)
    Z = np.dot(V[:n].T, Y[:n]).T

    axs[0].axis('equal')
    axs[0].axis('off')
    axs[0].imshow(np.reshape(X[selected], shape), cmap='Greys_r')

    axs[1].axis('equal')
    axs[1].axis('off')
    axs[1].imshow(np.reshape(Z, shape), cmap='Greys_r')

    fig.savefig(path_out, bbox_inches='tight', dpi=300)


def plot_mode_fraction(s, path_out):
    """Plots the fraction of power represented with n modes

    Args:
        s (array-like): 1D arrray of the variances of the principal components.
        path_out (str): Path to save figure to
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(0, 500)
    ax.set_ylim(1e-8, 5e-2)
    ax.set_yscale('log')
    ax.grid()
    f = 1 - np.cumsum(s**2) / np.sum(s**2)

    ax.set_xlabel('Modes')
    ax.set_ylabel('Missing mode power')

    ax.plot(f)
    fig.savefig(path_out, bbox_inches='tight')
