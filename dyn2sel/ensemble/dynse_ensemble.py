from dyn2sel.ensemble import Ensemble
from copy import deepcopy

import numpy as np


class DYNSEEnsemble(Ensemble):
    def __init__(self, clf, max_size=-1):
        self.clf = clf
        self.ensemble = []
        self.max_size = max_size

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        clf_copy = deepcopy(self.clf)
        clf_copy.partial_fit(X, y)
        self.add_member(clf_copy)

    def predict(self, X):
        predictions = np.empty((len(self.ensemble), X.shape[0]))
        for index_clf, clf in enumerate(self.ensemble):
            predictions[index_clf] = clf.predict(X)
        return predictions.T

    def predict_proba(self, X):
        pass

    def add_member(self, clf):
        self.ensemble.append(clf)

    def del_member(self, index=-1):
        self.ensemble.pop(index)