import numpy as np
import matplotlib.pyplot as plt


def plot_mode_proj(X, V, labels, modes):
    """Creates a 3D projection of X into the 3 selected SVD modes

    Args:
        X (array_like): Data matrix with rows of images
        V (array_like): Matrix with mode vectors as columns
        modes (list): List of 3 mode indexes to project on
    """
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection='3d')

    Y = np.dot(V[modes], X.T)
    ax.set_xlabel(f'Mode {modes[0]}')
    ax.set_ylabel(f'Mode {modes[1]}')
    ax.set_zlabel(f'Mode {modes[2]}')
    ax.scatter(Y[0], Y[1], Y[2], c=labels, s=1)

    fig.savefig('HW4/figures/svd_projection.png', bbox_inches='tight', dpi=300)


def plot_n_modes(X, V, n):
    """Shows the numbers represented with the selected number of SVD modes

    Args:
        X (array_like): Data matrix with rows of images
        V (array_like): Matrix with mode vectors as columns
        n (int): Number of modes to use in the representation
    """
    fig, axs = plt.subplots(4, 4, figsize=(4, 3))

    selected = np.random.randint(0, X.shape[0], 8)
    Y = np.dot(V, X[selected].T)
    Z = np.dot(V[:n].T, Y[:n]).T

    for ax, image in zip(axs[:2].flatten(), X[selected]):
        ax.axis('equal')
        ax.axis('off')
        ax.imshow(np.reshape(image, (28, 28)), cmap='Greys')

    for ax, image in zip(axs[2:].flatten(), Z):
        ax.axis('equal')
        ax.axis('off')
        ax.imshow(np.reshape(image, (28, 28)), cmap='Greys')
  
    fig.savefig('HW4/figures/reduced_dim.png', bbox_inches='tight', dpi=300)


def plot_svd_spectrum(X, V):
    """Plots the svd spectrum of 4 random images.

        Args:
        X (array_like): Data matrix with rows of images
        V (array_like): Matrix with mode vectors as columns
    """
    fig, axs = plt.subplots(2, 2, figsize=(6, 4))

    selected = np.random.randint(0, X.shape[0], 4)
    Y = np.dot(V, X[selected].T).T

    for ax, spectrum in zip(axs.flatten(), Y):
        ax.set_xlabel('Mode')
        ax.set_xlim(-1, 101)
        ax.plot(spectrum)

    fig.tight_layout()
    fig.savefig('HW4/figures/svd_spectrum.pdf', bbox_inches='tight')


def plot_mode_fraction(s):
    """Plots the fraction of power represented with n modes

    Args:
        s (array-like): 1D arrray of the variances of the principal components.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(0, 600)
    ax.set_ylim(1e-4,1)
    ax.set_yscale('log')
    ax.grid()
    f = 1 - np.cumsum(s**2) / np.sum(s**2)

    ax.set_xlabel('Modes')
    ax.set_ylabel('Missing mode power')

    ax.plot(f)
    fig.savefig('HW4/figures/mode_frac.pdf', bbox_inches='tight')
