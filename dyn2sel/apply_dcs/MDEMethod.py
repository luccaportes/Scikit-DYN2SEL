from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.ensemble import MDEEnsemble
from dyn2sel.dcs_techniques.mde_selection import MDESel

import numpy as np

class MDEMethod(DCSApplier):
    def __init__(self, clf, chunk_size, max_ensemble_size=-1, alpha=0.3):
        self.clf = clf
        self.chunk_size = chunk_size
        self.max_ensemble_size = max_ensemble_size
        self.val_set = None
        self.dcs_method = None
        self.ensemble = MDEEnsemble(clf, alpha=alpha)
        self.temp_buffer_x = []
        self.temp_buffer_y = []
        self.minority_class = -1

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        for x_i, y_i in zip(X, y):
            if len(self.temp_buffer_x) < self.chunk_size:
                self.temp_buffer_x.append(x_i)
                self.temp_buffer_y.append(y_i)
            else:
                self.ensemble.partial_fit(np.array(self.temp_buffer_x), np.array(self.temp_buffer_y))
                self.ensemble.classes_ = np.sort(np.unique(self.temp_buffer_y))
                self.minority_class = self.get_minority_class(self.temp_buffer_y)
                self.dcs_method = MDESel(self.minority_class)
                self.temp_buffer_x = []
                self.temp_buffer_y = []

    def predict(self, X, y=None):
        if len(self.ensemble) > 0:
            predictions = self.dcs_method.predict(self.ensemble, y)
            return predictions
        else:
            return np.array([])

    def predict_proba(self, X):
        pass

    def get_minority_class(self, y):
        unique_y, counts_y = np.unique(y, return_counts=True)
        return unique_y[np.argmin(counts_y)]
