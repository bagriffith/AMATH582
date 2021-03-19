import numpy as np
import matplotlib.pyplot as plt


def find_weights(X, b):
    """Calculate the weights of the linear regression.

    Args:
        X (array-like): Matrix with rows of curves.
        b (array-like): The capacity for each curve.
    Returns:
        ndarray: Weights for the model.
    """
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    w, _, _, _ = np.linalg.lstsq(X, b)
    return w


def predict(X, w):
    """Using previously calculated weights, predict the capacity of each curve.

    Args:
        X (array-like): Matrix with rows of curves.
        w (array-like): Weights for the model.

    Returns:
        ndarray: Predicted Capacities
    """
    X = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    return np.dot(X, w)


def accuracy_dist(b_real, b_model, path_out):
    """Plots the error for two sets of capacties

    Args:
        b_real (array-like): The capacity actually measured.
        b_model (array-like): Predicted capacities.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel(r'$R_{i,model} - R_i$')
    ax.set_ylabel('Number of Curves')

    error = b_model - b_real
    print(np.std(error))
    ax.hist(error, bins=24)
    fig.savefig(path_out, bbox_inches='tight')
