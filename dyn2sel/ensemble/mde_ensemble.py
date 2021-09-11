from copy import deepcopy

import numpy as np
from dyn2sel.ensemble import Ensemble
from dyn2sel.ensemble._skmultiflow_encapsulator import skmultiflow_encapsulator
from dyn2sel.utils import BalancedAccuracyEvaluator
from sklearn.neighbors import KNeighborsClassifier
from skmultiflow.core import ClassifierMixin


class MDEEnsemble(Ensemble):
    def __init__(self, clf, max_size=10, n_neighbors=3, alpha=0.3):
        super().__init__()
        self.clf = clf
        self.max_size = max_size
        self.n_neighbors = n_neighbors
        self.bac_ensemble = []
        self.n_instances_ensemble = []
        self.alpha = alpha

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.update_bac(X, y)
        self.remove_low_bac()
        clf_copy = deepcopy(self.clf)
        if issubclass(type(clf_copy), ClassifierMixin):
            clf_copy = skmultiflow_encapsulator(clf_copy)
        X, y = self.filter_outliers(X, y)
        clf_copy.partial_fit(X, y)
        if len(self.ensemble) >= self.max_size:
            self.del_member(self.get_worst_bac())
        self.add_member(clf_copy)

    def add_member(self, clf):
        self.ensemble.append(clf)
        self.bac_ensemble.append(BalancedAccuracyEvaluator())

    def del_member(self, index=-1):
        self.ensemble.pop(index)
        self.bac_ensemble.pop(index)

    def filter_outliers(self, X, y):
        knn = KNeighborsClassifier(n_neighbors=self.n_neighbors + 1).fit(X, y)
        unique_y, counts_y = np.unique(y, return_counts=True)
        minority_label, majority_label = (
            unique_y[np.argmin(counts_y)],
            unique_y[np.argmax(counts_y)],
        )
        minority_indexes = np.argwhere(y == minority_label).flatten()
        minority_X = X[minority_indexes]
        neighbors_index = knn.kneighbors(minority_X, return_distance=False)[:, 1:]
        neighbors_y = y[neighbors_index]
        with_maj_neighbors = neighbors_y == majority_label
        outliers = np.sum(with_maj_neighbors, axis=1) == self.n_neighbors
        if np.any(outliers):
            to_remove = minority_indexes[outliers]
            X = np.delete(X, to_remove, axis=0)
            y = np.delete(y, to_remove)
        return X, y

    def update_bac(self, X, y):
        for i, _ in enumerate(self.ensemble):
            self.bac_ensemble[i].add_results(y, self.ensemble[i].predict(X))

    def get_worst_bac(self):
        return np.argmin([i.get_bac() for i in self.bac_ensemble])

    def remove_low_bac(self):
        bacs = [i.get_bac() for i in self.bac_ensemble]
        to_remove = np.argwhere(np.array(bacs) < (0.5 + self.alpha)).flatten()
        for i in to_remove:
            self.del_member(i)
