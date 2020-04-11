from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.validation_set import ValidationSet
from dyn2sel.ensemble import DYNSEEnsemble

import numpy as np
import numpy.ma as ma


class DYNSEMethod(DCSApplier):
    def __init__(self, clf, chunk_size, dcs_method):
        self.clf = clf
        self.chunk_size = chunk_size
        # self.n_chunks = n_chunks
        self.dcs_method = dcs_method
        self.val_set = ValidationSet()
        self.ensemble = DYNSEEnsemble(clf)
        self.temp_buffer_x = []
        self.temp_buffer_y = []

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        for x_i, y_i in zip(X, y):
            if len(self.temp_buffer_x) < self.chunk_size:
                self.temp_buffer_x.append(x_i)
                self.temp_buffer_y.append(y_i)
            else:
                self.ensemble.partial_fit(np.array(self.temp_buffer_x), np.array(self.temp_buffer_y))
                self.val_set.replace_set(self.temp_buffer_x, self.temp_buffer_y)
                self.dcs_method.fit(np.array(self.temp_buffer_x), np.array(self.temp_buffer_y))
                self.temp_buffer_x = []
                self.temp_buffer_y = []

    def predict(self, X):
        if len(self.ensemble) > 0:
            all_predictions = self.ensemble.predict(X)
            selected_indexes = self.dcs_method.estimate_competence(self.ensemble, X)
            masked_predictions = ma.masked_array(all_predictions, selected_indexes)
            final_predictions = np.max(masked_predictions, axis=1)
            return final_predictions
        else:
            return np.array([])

    def predict_proba(self, X):
        pass