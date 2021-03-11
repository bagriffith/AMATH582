import numpy as np
from sklearn import svm, tree
import loadmnist
import svd
import evaluation
import lda


def run_analysis():
    """Runs the full analysis for the MNIST handwritting project"""
    X_large, labels_l = loadmnist.load_data()
    X_small, labels_s = loadmnist.load_data(size=10000)
    U, s, V = np.linalg.svd(X_small)
    V = V[:100]

    svd.plot_mode_proj(X_large, V, labels_l, [1, 2, 3])
    svd.plot_n_modes(X_large, V, 100)
    svd.plot_svd_spectrum(X_large, V)
    svd.plot_mode_fraction(s)

    N = 2000
    evaluation.number_confusion(evaluation.NaiveClassifier(), N, V)
    evaluation.number_confusion(lda.LDA(), N, V)
    evaluation.number_confusion(svm.SVC(), N, V)
    evaluation.number_confusion(tree.DecisionTreeClassifier(), N, V)

    N = 3000
    print('Good:')
    print(evaluation.digit_performance(lda.LDA(), N, V, [0, 2, 8]))
    print('Bad:')
    print(evaluation.digit_performance(lda.LDA(), N, V, [0, 4, 5]))

    N = 20000
    evaluation.full_classification(svm.SVC(), N, V)
    evaluation.full_classification(tree.DecisionTreeClassifier(), N, V)


if __name__ == '__main__':
    run_analysis()
