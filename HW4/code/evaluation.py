import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import combinations
import loadmnist


class NaiveClassifier:
    """A model classifier that randomly guesses a digit.

    This was created as a simple test article to make sure the number_confusion
    code worked indpendent of any model used.
    """
    def fit(self, X, y):
        self.choices = np.int8(list(set(y)))

    def predict(self, X):
        index = np.random.randint(0, len(self.choices), X.shape[0])
        return self.choices[index]


def number_confusion(model, train_n, V):
    """Plots a how the model preforms at distinguising pairs of digits.

    Args:
        model: The model class. Should have functions fit(X, y) that trains the 
            model to identify labels y using data X and predict(X) that will
            label data in the matrix X.
        train_n (int): The number of example digits to train on.
        V (array-like): A matrix to transform the data into the basis for
            predictions.
    """
    error_rate = np.full((10, 10), np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))

    for digits in combinations(range(10), 2):
        X, labels = loadmnist.load_data(numbers=digits,
                                        size=train_n+(train_n//5))

        Y = np.dot(V, X.T).T

        model.fit(Y[:train_n], labels[:train_n])
        model_lables = model.predict(Y[train_n:])

        errors = np.sum(model_lables != labels[train_n:])
        e = 100*errors / len(model_lables)
        error_rate[digits[1], digits[0]] = e
        ax.text(digits[0], digits[1], f'{e:0.1f}', c='k',
                va='center', ha='center')

    ax.set_xlim(-.5, 8.5)
    ax.set_ylim(.5, 9.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    X, Y = np.meshgrid(np.arange(11)-.5, np.arange(11)-.5)
    m = np.nanmean(error_rate)
    r = np.nanmax(np.abs(error_rate - m))

    mesh = ax.pcolormesh(X, Y, error_rate, cmap='coolwarm', vmin=m-r, vmax=m+r)
    l = np.floor(10*np.nanmin(error_rate))/10
    r = np.ceil(10*np.nanmax(error_rate))/10
    bounds = np.linspace(l, r, 512)
    cbar = fig.colorbar(mesh, boundaries=bounds, label='Error Rate %')
    cbar.set_ticks(MaxNLocator(8))
    fig.savefig('HW4/figures/{}-digits_conf.pdf'.format(type(model).__name__),
                bbox_inches='tight')


def full_classification(model, train_n, V):
    """Plots a how the model preforms at identifying digits.

    Args:
        model: The model class. Should have functions fit(X, y) that trains the 
            model to identify labels y using data X and predict(X) that will
            label data in the matrix X.
        train_n (int): The number of example digits to train on.
        V (array-like): A matrix to transform the data into the basis for
            predictions.
    """
    frac_rate = np.full((10, 10), np.nan)

    fig, ax = plt.subplots(figsize=(6, 4))

    X, labels = loadmnist.load_data(size=train_n+(train_n//5))

    Y = np.dot(V, X.T).T
    model.fit(Y[:train_n], labels[:train_n])
    model_lables = model.predict(Y[train_n:])

    ax.set_xlabel('Real #')
    ax.set_ylabel('Predicted #')

    total = np.bincount(labels[train_n:])

    for real in range(10):
        for predict in range(10):
            n = np.sum((model_lables == predict) & (labels[train_n:] == real))
            frac = 100*n / total[real]
            frac_rate[real, predict] = frac
            ax.text(real, predict, f'{frac:0.1f}', c='k',
                    va='center', ha='center')

    ax.set_xlim(-.5, 9.5)
    ax.set_ylim(-.5, 9.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    X, Y = np.meshgrid(np.arange(11)-.5, np.arange(11)-.5)

    off_diag = frac_rate.copy()
    np.fill_diagonal(off_diag, np.nan)
    ax.pcolormesh(X, Y, off_diag, cmap='YlOrRd', alpha=.4)

    on_diag = np.full_like(off_diag, np.nan)
    np.fill_diagonal(on_diag, 1)
    on_diag *= frac_rate
    m = np.nanmin(on_diag) - (np.nanmax(on_diag)-np.nanmin(on_diag))*.5
    ax.pcolormesh(X, Y, on_diag, cmap='Greens', vmin=m)

    fig.savefig('HW4/figures/{}-classification.pdf'.format(type(model).__name__),
                bbox_inches='tight')
    

def digit_performance(model, train_n, V, digits):
    """Plots a how the model preforms at identifying digits.

    Args:
        model: The model class. Should have functions fit(X, y) that trains the 
            model to identify labels y using data X and predict(X) that will
            label data in the matrix X.
        train_n (int): The number of example digits to train on.
        V (array-like): A matrix to transform the data into the basis for
            predictions.
        digits (list): List of digits to test on
    """
    X, labels = loadmnist.load_data(numbers=digits,
                                    size=train_n+(train_n//5))

    Y = np.dot(V, X.T).T

    model.fit(Y[:train_n], labels[:train_n])
    model_lables = model.predict(Y[train_n:])

    errors = np.sum(model_lables != labels[train_n:])
    error_rate = 100*errors / len(model_lables)
    return error_rate
