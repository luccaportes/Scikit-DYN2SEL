from dyn2sel.ensemble import Ensemble
from copy import deepcopy
from skmultiflow.core import ClassifierMixin
from skmultiflow.metrics import ClassificationPerformanceEvaluator
from dyn2sel.ensemble._skmultiflow_encapsulator import skmultiflow_encapsulator

import numpy as np


class DDCSEnsemble(Ensemble):
    def __init__(self, clf, max_size=-1, use_bagging=False, init_all=False):
        super().__init__()
        self.clf = clf
        self.max_size = max_size
        self.use_bagging = use_bagging
        self.init_all = init_all
        self.acc_ensemble = []

    def _init_ensemble(self):
        if self.max_size == -1:
            raise ValueError("init_all can not be true if max_size == -1")
        for _ in range(self.max_size):
            clf_copy = deepcopy(self.clf)
            if issubclass(type(clf_copy), ClassifierMixin):
                clf_copy = skmultiflow_encapsulator(clf_copy)
            self.add_member(clf_copy)

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        clf_copy = deepcopy(self.clf)
        if issubclass(type(clf_copy), ClassifierMixin):
            clf_copy = skmultiflow_encapsulator(clf_copy)
        self.add_member(clf_copy)
        if len(self.ensemble) > self.max_size != -1:
            self.del_member(self.get_worst_acc())
        for i in self.ensemble:
            if self.use_bagging:
                poiss_lambda = np.random.poisson(1, X.shape[0])
                X = np.repeat(X, poiss_lambda, axis=0)
                y = np.repeat(y, poiss_lambda)
            i.partial_fit(X, y, classes=classes, sample_weight=sample_weight)
        self.update_accuracies(X, y)

    def update_accuracies(self, X, y):
        for index_ensemble, _ in enumerate(self.ensemble):
            preds = self.ensemble[index_ensemble].predict(X)
            for index_pred in range(y.shape[0]):
                self.acc_ensemble[index_ensemble].add_result(
                    y[index_pred], preds[index_pred]
                )

    def add_member(self, clf):
        self.ensemble.append(clf)
        self.acc_ensemble.append(ClassificationPerformanceEvaluator())

    def del_member(self, index=0):
        self.ensemble.pop(index)
        self.acc_ensemble.pop(index)

    def get_worst_acc(self):
        return np.argmin([i.accuracy_score() for i in self.acc_ensemble])
