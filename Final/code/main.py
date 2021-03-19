import numpy as np
import matplotlib.pyplot as plt
import loadCurves
import svd
import predict


def plot_aging(p, X, labels, capacity):
    """Plot examples of new and old battery curves.

    Args:
        X (ndarray): Matrix of curves
        p (ndarray): The discharge percentage that X rows are a function of
        labels (ndarray): Whuch battery each curve is
        capacity (ndarray): The capacity of each battery during each curve
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel('Energy Discharged')
    ax.set_ylabel('Battery Voltage')

    bat_sel = labels == 5
    curve_new = X[bat_sel][np.argmax(capacity[bat_sel])]
    curve_old = X[bat_sel][np.argmin(capacity[bat_sel])]
    ax.plot(p, curve_new, label='New')
    ax.plot(p, curve_old, label='EOL')
    ax.legend()
    fig.savefig('Final/figures/aging_curves.pdf', bbox_inches='tight')


def run_analysis():
    loadCurves.create_nasa_curves()
    X, p, labels, capacity = loadCurves.load_nasa_curves([5, 6, 7])
    plot_aging(p, X, labels, capacity)

    u, s, vh = np.linalg.svd(X)
    svd.plot_mode_fraction(s)
    k = 20
    svd.plot_n_modes(p, X, vh, k)
    svd.plot_mode_vs_life(capacity, X, vh, 6)

    Y = np.dot(vh, X.T)[:k].T
    w = predict.find_weights(Y, capacity)
    print(w)
    cap_pred = predict.predict(Y, w)
    predict.accuracy_dist(capacity, cap_pred, 'Final/figures/train-pref.pdf')

    X_tst, _, _, cap_tst = loadCurves.load_nasa_curves([18])
    Y = np.dot(vh, X_tst.T)[:k].T
    cap_pred = predict.predict(Y, w)
    predict.accuracy_dist(cap_tst, cap_pred, 'Final/figures/test-pref.pdf')


if __name__ == '__main__':
    run_analysis()
