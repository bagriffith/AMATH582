import numpy as np
import matplotlib.pyplot as plt


def plot_mode_fraction(s):
    """Plots the fraction of power represented with n modes.

    Args:
        s (array-like): 1D arrray of the variances of the principal components.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlim(0, 30)
    ax.set_ylim(1e-8, 1)
    ax.set_yscale('log')
    ax.grid()
    f = 1 - np.cumsum(s**2) / np.sum(s**2)

    ax.set_xlabel('Modes')
    ax.set_ylabel('Missing mode power')

    ax.plot(f, marker='o')
    fig.savefig('Final/figures/mode_frac.pdf', bbox_inches='tight')


def plot_n_modes(p, X, V, n):
    """Shows the numbers represented with the selected number of SVD modes.

    Args:
        p (array_like): The discharge percentage that X rows are a function of.
        X (array_like): Data matrix with rows of images.
        V (array_like): Matrix with mode vectors as columns.
        n (int): Number of modes to use in the representation.
    """
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlabel('Energy Discharged')
    ax.set_ylabel('Battery Voltage')
    
    selected = -1
    Y = np.dot(V, X[selected])
    Z = np.dot(V[:n].T, Y[:n]).T

    ax.plot(p, X[selected], label='Actual')
    ax.plot(p, Z, label='Red. dim.')
    ax.legend()
    fig.savefig('Final/figures/red_dim.pdf', bbox_inches='tight')


def plot_mode_vs_life(capacity, X, V, n):
    """Plots how the capacity changes for values of each SVD mode.

    Args:
        capacity (ndarray): The capacity of each battery during each curve.
        X (ndarray): Matrix of curves.
        V (array_like): Matrix with mode vectors as columns.
        n (int): Number of modes to plot.
    """
    Y = np.dot(V, X.T)[:n]

    fig, axs = plt.subplots(int(np.ceil(n/3)), 3, figsize=(10, 6))

    for n, (C, ax) in enumerate(zip(Y, axs.flatten())):
        ax.set_xlabel(f'Mode {n}')
        ax.set_ylabel('Capacity Remaining')
        t = .15
        l = np.quantile(C, t)
        r = np.quantile(C, 1-t)
        ax.set_xlim(l - .2*(r-l), r + .2*(r-l))
        ax.scatter(C, capacity, s=.1)
    
    fig.tight_layout()
    fig.savefig('Final/figures/mode_proj.pdf', bbox_inches='tight')

