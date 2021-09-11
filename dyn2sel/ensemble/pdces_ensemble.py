from copy import deepcopy

import numpy as np
from dyn2sel.ensemble import Ensemble
from dyn2sel.utils import BalancedAccuracyEvaluator
from scipy import stats
from sklearn.utils import shuffle


class StratifiedBagging:
    def __init__(self, clf, size):
        self.clf = clf
        self.size = size
        self.ensemble = None
        self.classes_ = None

    def fit(self, X, y, classes=None):
        if self.classes_ is None:
            self.classes_ = classes
        self.ensemble = [deepcopy(self.clf) for i in range(self.size)]
        for index, _ in enumerate(self.ensemble):
            new_X = []
            new_y = []
            for j in np.unique(y):
                filtered_x = X[y == j]
                stratified_bag = list(
                    filtered_x[
                        np.random.choice(
                            filtered_x.shape[0], len(filtered_x), replace=True
                        ),
                        :,
                    ]
                )
                new_X += stratified_bag
                new_y += [j] * len(stratified_bag)
            new_X, new_y = shuffle(new_X, new_y)
            self.ensemble[index].partial_fit(
                np.array(new_X), np.array(new_y), self.classes_
            )
        return self

    def partial_fit(self, X, y, classes=None):
        return self.fit(X, y, classes)

    def predict(self, X):
        if self.ensemble is not None:
            preds = np.array([i.predict(X) for i in self.ensemble])
            final_preds, _ = stats.mode(preds, axis=0)
            return final_preds.reshape(
                -1,
            )
        return np.array([])


class PDCESEnsemble(Ensemble):
    def __init__(self, clf, max_size=10, bagging_size=5):
        super().__init__()
        self.clf = clf
        self.max_size = max_size if max_size > 0 else float("inf")
        self.bac_ensemble = []
        self.bagging_size = bagging_size

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.update_bac(X, y)
        bag = StratifiedBagging(deepcopy(self.clf), self.bagging_size)
        # if issubclass(type(clf_copy), ClassifierMixin):
        #     clf_copy = skmultiflow_encapsulator(clf_copy)
        bag.partial_fit(X, y, classes=classes)
        if len(self.ensemble) >= self.max_size:
            self.del_member(self.get_worst_bac())
        self.add_member(bag)

    def add_member(self, clf):
        self.ensemble.append(clf)
        self.bac_ensemble.append(BalancedAccuracyEvaluator())

    def del_member(self, index=-1):
        self.ensemble.pop(index)
        self.bac_ensemble.pop(index)

    def update_bac(self, X, y):
        for i in range(len(self.ensemble)):
            self.bac_ensemble[i].add_results(y, self.ensemble[i].predict(X))

    def get_worst_bac(self):
        return np.argmin([i.get_bac() for i in self.bac_ensemble])
