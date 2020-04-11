from dyn2sel.apply_dcs import DCSApplier

import numpy as np
import numpy.ma as ma
import abc
import scipy.stats
from collections import defaultdict


class BaseMetric(abc.ABC):
    @abc.abstractmethod
    def get_value(self, predicted_class):
        pass

    @abc.abstractmethod
    def update_value(self, predicted_class, real_class):
        pass


class AccuracyMetric(BaseMetric):
    def __init__(self, window_size):
        self.window_size = window_size
        self.last_results = np.zeros(window_size)
        self.index = 0
        self.numerator = 0
        self.denominator = 0
        self.first_run = True

    def get_value(self, predicted_class):
        if not self.first_run:
            return self.numerator / self.denominator
        else:
            if self.denominator > (self.window_size / 4):
                return self.numerator / self.denominator
            return -1

    def update_value(self, predicted_class, real_class):
        pred_result = int(predicted_class == real_class)
        replaced_result = self.last_results[self.index]
        self.last_results[self.index] = pred_result
        if self.index + 1 != self.window_size:
            self.index += 1
        else:
            self.index = 0
        self.numerator += pred_result - replaced_result
        if self.denominator != self.window_size:
            self.denominator += 1


class RecallMetric(BaseMetric):
    def __init__(self, window_size):
        self.window_size = window_size
        self.last_results = defaultdict(lambda: np.zeros(window_size))
        self.index = defaultdict(int)
        self.numerator = defaultdict(int)
        self.denominator = defaultdict(int)
        self.first_run = defaultdict(lambda: True)

    def get_value(self, predicted_class):
        if not self.first_run[predicted_class]:
            return self.numerator[predicted_class] / self.denominator[predicted_class]
        else:
            self.first_run[predicted_class] = False
            if self.denominator[predicted_class] > (self.window_size / 4):
                return self.numerator[predicted_class] / self.denominator[predicted_class]
            return -1

    def update_value(self, predicted_class, real_class):
        pred_result = int(predicted_class == real_class)
        replaced_result = self.last_results[real_class][self.index[real_class]]
        self.last_results[real_class][self.index[real_class]] = pred_result
        if self.index[real_class] + 1 != self.window_size:
            self.index[real_class] += 1
        else:
            self.index[real_class] = 0
        self.numerator[real_class] += pred_result - replaced_result
        if self.denominator[real_class] != self.window_size:
            self.denominator[real_class] += 1


class WPSMethod(DCSApplier):
    def __init__(self, clf, n_selected, window_size=2000, metric="accuracy"):
        self.clf = clf
        self.n_selected = n_selected
        self.metrics = self._build_metric(metric, window_size, clf.n_estimators)

    @staticmethod
    def _build_metric(metric, window_size, ensemble_size):
        if metric == "accuracy":
            return [AccuracyMetric(window_size) for _ in range(ensemble_size)]
        if metric == "recall":
            return [RecallMetric(window_size) for _ in range(ensemble_size)]

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.clf.partial_fit(X, y, classes, sample_weight)
        for clf_index in range(len(self.clf.ensemble)):
            predictions = self.clf.ensemble[clf_index].predict(X)
            for pred_index in range(len(predictions)):
                self.metrics[clf_index].update_value(predictions[pred_index], y[pred_index])
        return self

    def predict(self, X):
        if self.clf.ensemble is not None:
            predictions = np.array([i.predict(X) for i in self.clf.ensemble], dtype=np.float)
            metrics = np.empty(predictions.shape)
            for clf_index in range(len(self.clf.ensemble)):
                metrics[clf_index] = np.array([self.metrics[clf_index].get_value(i) for i in predictions[clf_index]])
            if np.any(metrics == -1):
                return self.clf.predict(X)
            ranking = np.argsort(-metrics, axis=0)
            not_selected = ranking >= self.n_selected
            predictions[not_selected] = np.nan
            final_predictions = scipy.stats.mode(predictions, axis=0)[0][0].astype(np.int)
            return final_predictions
        else:
            return self.clf.predict(X)


    def predict_proba(self, X):
        pass


class WPSMethodPostDrift(DCSApplier):
    def __init__(self, clf, n_selected, window_size=2000, metric="accuracy"):
        self.clf = clf
        self.n_selected = n_selected
        self.metrics = self._build_metric(metric, window_size, clf.n_estimators)

    @staticmethod
    def _build_metric(metric, window_size, ensemble_size):
        if metric == "accuracy":
            return [AccuracyMetric(window_size) for _ in range(ensemble_size)]
        if metric == "recall":
            return [RecallMetric(window_size) for _ in range(ensemble_size)]

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.clf.partial_fit(X, y, classes, sample_weight)
        for clf_index in range(len(self.clf.ensemble)):
            predictions = self.clf.ensemble[clf_index].predict(X)
            for pred_index in range(len(predictions)):
                self.metrics[clf_index].update_value(predictions[pred_index], y[pred_index])
        return self

    def predict(self, X):
        if self.clf.ensemble is not None:
            predictions = np.array([i.predict(X) for i in self.clf.ensemble], dtype=np.float)
            metrics = np.empty(predictions.shape)
            for clf_index in range(len(self.clf.ensemble)):
                metrics[clf_index] = np.array([self.metrics[clf_index].get_value(i) for i in predictions[clf_index]])
            if np.any(metrics == -1):
                return self.clf.predict(X)
            ranking = np.argsort(-metrics, axis=0)
            not_selected = ranking >= self.n_selected
            predictions[not_selected] = np.nan
            final_predictions = scipy.stats.mode(predictions, axis=0)[0][0].astype(np.int)
            return final_predictions
        else:
            return self.clf.predict(X)

    def predict_proba(self, X):
        pass



