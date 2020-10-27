from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np
from collections import defaultdict


class BalancedAccuracyEvaluator:
    """
    BalancedAccuracyEvaluator is a class that implements an online computation
    of the balanced_accuracy_score present is scikit-learn. The way to compute it
    is based on the way it was implemented in its original version on scikit-learn.
    """

    def __init__(self):
        self._dict_hits = defaultdict(int)
        self._dict_total = defaultdict(int)

    def add_results(self, y_true, y_pred):
        """
        Parameters
        ----------
        y_true : 1d array-like
            Ground truth (correct) target values.
        y_pred : 1d array-like
            Estimated targets as returned by a classifier.
        """
        for i, _ in enumerate(y_true):
            self._add_result(y_true[i], y_pred[i])

    def _add_result(self, y_true, y_pred):
        if y_true == y_pred:
            self._dict_hits[y_true] += 1
        self._dict_total[y_true] += 1

    def get_bac(self):
        diag = np.array(
            [self._dict_hits[key] for key in sorted(self._dict_total.keys())]
        )
        total = np.array(
            [self._dict_total[key] for key in sorted(self._dict_total.keys())]
        )
        per_class = diag / total
        return np.mean(per_class)
