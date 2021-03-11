import numpy as np
import itertools
from scipy.linalg import eig


class LDA:
    """Linear Discrimination Analysis model for classifying in up to 3 groups.
    """
    def fit(self, X, y):
        """Trains the model.

        Args:
            X (array-like): Contains rows of the training data examples.
            y (array-like): Contains labels for the training data rows.
        """
        self.digits = list(set(y))

        if len(self.digits) == 2:
            X0, X1 = (X[y == d] for d in self.digits)
            m0, m1 = (np.mean(Xi, axis=0) for Xi in [X0, X1])

            S_b = np.outer(m1 - m0, m1 - m0)

            S_w = np.zeros((len(m0), len(m0)))
            for m_j in [m0, m1]:
                for i in range(X.shape[0]):
                    S_w += np.outer(X[i] - m_j, X[i] - m_j)

            W, V = eig(S_b, S_w)
            i = np.argmax(W)
            w = V[:, i]
            self.wt = (w / np.linalg.norm(w)).T

            v0 = np.dot(X0, self.wt)
            v1 = np.dot(X1, self.wt)

            if np.median(v1) < np.median(v0):
                self.wt = -self.wt
                v0 = -v0
                v1 = -v1

            x = np.linspace(0, .5, 1024)
            p0 = np.quantile(v0, x)
            p1 = np.quantile(v0, 1-x)
            mp = np.argmin(np.abs(p1 - p0))
            self.threshold = (p0[mp] + p1[mp])/2
        elif len(self.digits) == 3:
            self.pair_ldas = []
            self.pair_digits = []
            for d_select in itertools.combinations(self.digits, 2):
                self.pair_ldas.append(LDA())
                self.pair_digits.append(d_select)
                mask = np.isin(y, d_select)
                self.pair_ldas[-1].fit(X[mask], y[mask])
        else:
            print(self.digits)
            raise RuntimeError()

    def predict(self, X):
        """Predicts the category of rows of X.

        Args:
            X (array-like): Contains rows of the data to categorize.
        """
        if len(self.digits) == 2:
            r = np.dot(X, self.wt)
            return np.where(r < self.threshold, self.digits[0], self.digits[1])
        if len(self.digits) == 3:
            votes = np.zeros((X.shape[0], 3))
            for i, (d_select, lda) in enumerate(zip(self.pair_digits, self.pair_ldas)):
                votes[:, i] = lda.predict(X)

            counts = np.zeros((X.shape[0], 3), dtype=np.int64)
            for i in range(votes.shape[0]):
                for j in range(3):
                    for k, d in enumerate(self.digits):
                        if votes[i, j] == d:
                            counts[i, k] += 1

            label = np.zeros(X.shape[0], np.int8)
            for i in range(X.shape[0]):
                if np.max(counts[i]) > 1:
                    label[i] = self.digits[np.argmax(counts[i])]
                else:
                    label[i] = self.digits[np.random.randint(3)]
        
            return label
        else:
            raise RuntimeError()
