from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.ensemble import PDCESEnsemble
from dyn2sel.dcs_techniques.mde_selection import MDESel
from dyn2sel.validation_set import ValidationSet
from dyn2sel.dcs_techniques import KNORAE
from imblearn.over_sampling import SMOTE
import copy

import numpy as np


class PDCESMethod(DCSApplier):
    """
    PDCESMethod
    
    Parameters
    ----------
    clf : Scikit-Multiflow Classifier
        The base classifier used for populating the ensemble

    chunk_size : integer
        The size of the chunks to accumulate data before fitting a classifier.

    max_ensemble_size : integer, default=-1
        The maximum size that an ensemble can grow. If -1, it grows indefinitely.

    alpha : float between 0 and 1, default=0.3
        Value that composes the threshold for removing classifiers with low Balanced Class Accuracy (BAC), If one
        classifier is with value less than 0.5 + alpha, it is removed from the ensemble.

    References
    ----------
        Zyblewski, P.; Ksieniewicz, P.; Woźniak, M. Classifier selection for highly imbalanced data streams with
        minority driven ensemble. In: Artificial Intelligence and Soft Computing. Cham: Springer International
        Publishing, 2019. p. 626–635. ISBN 978-3-030-20912-4
    """

    def __init__(self, clf, chunk_size, max_ensemble_size=-1, dcs_method=KNORAE(), preprocess=SMOTE()):
        self.clf = clf
        self.chunk_size = chunk_size
        self.max_ensemble_size = max_ensemble_size
        self.val_set = ValidationSet()
        self.dcs_method = dcs_method
        self.preprocess = preprocess
        self.ensemble = PDCESEnsemble(clf)
        self.temp_buffer_x = []
        self.temp_buffer_y = []

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        for x_i, y_i in zip(X, y):
            if len(self.temp_buffer_x) < self.chunk_size:
                self.temp_buffer_x.append(x_i)
                self.temp_buffer_y.append(y_i)
            else:
                self.ensemble.partial_fit(
                    np.array(self.temp_buffer_x), np.array(self.temp_buffer_y)
                )
                preproc_method = copy.deepcopy(self.preprocess)
                X_res, y_res = preproc_method.fit_resample(self.temp_buffer_x, self.temp_buffer_y)
                self.val_set.replace_set(X_res, y_res)
                self.dcs_method.fit(X_res, y_res)
                self.ensemble.classes_ = np.sort(np.unique(self.temp_buffer_y))
                self.temp_buffer_x = []
                self.temp_buffer_y = []

    def predict(self, X, y=None):
        if len(self.ensemble) > 0:
            predictions = self.dcs_method.predict(self.ensemble, X)
            return predictions
        else:
            return np.array([])

    def predict_proba(self, X):
        pass

    def get_minority_class(self, y):
        unique_y, counts_y = np.unique(y, return_counts=True)
        return unique_y[np.argmin(counts_y)]
