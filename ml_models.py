"""Part 3: Data-Driven Classification Models (kNN and CART Decision Tree) & Extrapolation Test."""

import numpy as np


class KNNClassifier:
    """Z-score standardized Euclidean k-Nearest Neighbours classifier."""
    def __init__(self, k=3):
        self.k = k
        self.mean_ = None
        self.std_ = None
        self.X_tr_ = None
        self.y_tr_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.where(np.std(X, axis=0) == 0, 1.0, np.std(X, axis=0))
        self.X_tr_ = (X - self.mean_) / self.std_
        self.y_tr_ = np.asarray(y, dtype=int)
        return self

    def predict(self, X):
        Xs = (np.asarray(X, dtype=float) - self.mean_) / self.std_
        dists = np.sqrt(np.sum((Xs[:, None, :] - self.X_tr_[None, :, :])**2, axis=2))
        nn_idx = np.argsort(dists, axis=1)[:, :self.k]
        nn_labels = self.y_tr_[nn_idx]
        return (np.sum(nn_labels, axis=1) > (self.k / 2.0)).astype(int)


class DecisionTreeClassifierScratch:
    """Binary CART Decision Tree classifier using Gini impurity."""
    def __init__(self, max_depth=2, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y == 1)
        return 1.0 - (p1**2 + (1.0 - p1)**2)

    def _build(self, X, y, depth):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return {"leaf": True, "pred": int(np.round(np.mean(y)))}

        best_gain, best_feat, best_thresh = -1.0, None, None
        parent_gini = self._gini(y)
        for feat in range(n_features):
            vals = np.unique(X[:, feat])
            if len(vals) <= 1:
                continue
            for thresh in 0.5 * (vals[:-1] + vals[1:]):
                mask = X[:, feat] <= thresh
                y_l, y_r = y[mask], y[~mask]
                if len(y_l) == 0 or len(y_r) == 0:
                    continue
                gain = parent_gini - (len(y_l) / n_samples) * self._gini(y_l) - (len(y_r) / n_samples) * self._gini(y_r)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, float(thresh)

        if best_feat is None or best_gain <= 0.0:
            return {"leaf": True, "pred": int(np.round(np.mean(y)))}

        mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": self._build(X[mask], y[mask], depth + 1),
            "right": self._build(X[~mask], y[~mask], depth + 1),
        }

    def fit(self, X, y):
        self.tree_ = self._build(np.asarray(X, dtype=float), np.asarray(y, dtype=int), 0)
        return self

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["pred"]
        return self._predict_one(x, node["left"] if x[node["feat"]] <= node["thresh"] else node["right"])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(row, self.tree_) for row in X], dtype=int)


def chronological_train_test_split(df_healthy, df_faulty, train_frac=0.70):
    """Splits each record chronologically: first 70% windows for training, last 30% for testing."""
    n_h = int(round(len(df_healthy) * train_frac))
    n_f = int(round(len(df_faulty) * train_frac))
    import pandas as pd
    train_df = pd.concat([df_healthy.iloc[:n_h], df_faulty.iloc[:n_f]], ignore_index=True)
    test_df = pd.concat([df_healthy.iloc[n_h:], df_faulty.iloc[n_f:]], ignore_index=True)
    return train_df, test_df
