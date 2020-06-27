from dyn2sel.ensemble import Ensemble
from copy import deepcopy
from skmultiflow.core import ClassifierMixin
from dyn2sel.ensemble._skmultiflow_encapsulator import skmultiflow_encapsulator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from skmultiflow.drift_detection import ADWIN
from skmultiflow.metrics import ClassificationPerformanceEvaluator

import numpy as np


class DESDDEnsemble(Ensemble):
    def __init__(self, base_ensemble, ensemble_size, min_lambda, max_lambda):
        super().__init__()
        self.base_ensemble = base_ensemble
        self.ensemble_size = ensemble_size
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.lambdas = None
        self.accuracies = None

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if len(self.ensemble) == 0:
            self.init_ensemble()
        self.update_accuracy(X, y)
        for index, ens in enumerate(self.ensemble):
            poiss_lambda = np.random.poisson(self.lambdas[index], X.shape[0])
            temp_x = np.repeat(X, poiss_lambda, axis=0)
            temp_y = np.repeat(y, poiss_lambda)
            ens.partial_fit(temp_x, temp_y)

    def predict(self, X):
        predictions = np.empty((len(self.ensemble), X.shape[0]))
        for index_clf, clf in enumerate(self.ensemble):
            predictions[index_clf] = clf.predict(X)
        return predictions.T

    def predict_proba(self, X):
        pass

    def add_member(self, clf):
        pass

    def del_member(self, index=-1):
        pass

    def init_ensemble(self):
        self.ensemble = [deepcopy(self.base_ensemble) for _ in range(self.ensemble_size)]
        self.lambdas = np.random.randint(self.min_lambda, self.max_lambda+1, self.ensemble_size)
        self.accuracies = [ClassificationPerformanceEvaluator() for _ in range(self.ensemble_size)]

    def clear_ensemble(self):
        self.ensemble = []
        self.lambdas = None
        self.accuracies = None

    def update_accuracy(self, X, y):
        for index, ens in enumerate(self.ensemble):
            preds = ens.predict(X)
            for i in range(preds.shape[0]):
                self.accuracies[index].add_result(y[i], preds[i])

    def get_max_accuracy(self):
        return np.argmin([i.accuracy_score() for i in self.accuracies])



