from copy import deepcopy

import numpy as np
from dyn2sel.ensemble import Ensemble
from dyn2sel.ensemble._skmultiflow_encapsulator import skmultiflow_encapsulator
from dyn2sel.utils import BalancedAccuracyEvaluator
from skmultiflow.core import ClassifierMixin


class DPDESEnsemble(Ensemble):
    def __init__(self, clf, max_size=10, alpha=0.3):
        super().__init__()
        self.clf = clf
        self.max_size = max_size
        self.bac_ensemble = []
        self.alpha = alpha

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.update_bac(X, y)
        clf_copy = deepcopy(self.clf)
        if issubclass(type(clf_copy), ClassifierMixin):
            clf_copy = skmultiflow_encapsulator(clf_copy)
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
