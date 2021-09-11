from dyn2sel.apply_dcs.base import DCSApplier
from dyn2sel.ensemble import DPDESEnsemble
from dyn2sel.dcs_techniques import KNORAE
from imblearn.over_sampling import SMOTE
import copy

import numpy as np


class DPDESMethod(DCSApplier):
    """
    PDCESMethod
    The Preprocess Dynamic Classsifier Ensemble Selection (PDCES) is not only a selection method but a whole framework
    that covers training an ensemble with a data stream and predicting new instances. It focuses on data that are
    affected by the imbalanced class problem. This is done by replacing poor performing classifiers with new ones.
    The method divides the stream into chunks of data with a fixed size. Each chunk is passed as a mini-batch for the
    ensemble to train on. Each data chunk is firstly sent to be predicted, then to train a new classifier and add it
    to the ensemble. The prediction step is performed using traditional DCS methods, with a validation set that is
    defined as the last trained chunk. As this method is focused on imbalanced problems, before training and updating
    the validation set, a preprocessing (over/undersampling) step is performed on the chunk.

    Parameters
    ----------
    clf : Scikit-Multiflow Classifier
        The base classifier used for populating the ensemble

    chunk_size : integer
        The size of the chunks to accumulate data before fitting a classifier.

    max_ensemble_size : integer, default=-1
        The maximum size that an ensemble can grow. If -1, it grows indefinitely.

    dcs_method : DCSTechnique object
        Dynamic selection technique to be used in the prediction process.

    preprocess : SamplerMixin object from imblearn
        Preprocess method to use when updating the validation set

    References
    ----------
        Zyblewski, P., Sabourin, R., & Woźniak, M. (2021). Preprocessed dynamic classifier ensemble selection for
        highly imbalanced drifted data streams. Information Fusion, 66, 138–154.
        https://doi.org/10.1016/j.inffus.2020.09.004
    """

    def __init__(
        self,
        clf,
        chunk_size,
        max_ensemble_size=-1,
        dcs_method=KNORAE(),
        preprocess=SMOTE(),
    ):
        self.clf = clf
        self.chunk_size = chunk_size
        self.max_ensemble_size = max_ensemble_size
        self.dcs_method = dcs_method
        self.preprocess = preprocess
        self.ensemble = DPDESEnsemble(clf)
        self.temp_buffer_x = []
        self.temp_buffer_y = []

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        for x_i, y_i in zip(X, y):
            if len(self.temp_buffer_x) < self.chunk_size:
                self.temp_buffer_x.append(x_i)
                self.temp_buffer_y.append(y_i)
            else:
                preproc_method = copy.deepcopy(self.preprocess)
                X_res, y_res = preproc_method.fit_resample(
                    self.temp_buffer_x, self.temp_buffer_y
                )
                self.ensemble.partial_fit(X_res, y_res)
                # self.val_set.replace_set(X_res, y_res)
                self.dcs_method.fit(X_res, y_res)
                self.ensemble.classes_ = np.sort(np.unique(self.temp_buffer_y))
                self.temp_buffer_x = []
                self.temp_buffer_y = []

    def predict(self, X, y=None):
        if len(self.ensemble) > 0:
            predictions = self.dcs_method.predict(self.ensemble, X)
            return predictions
        return np.array([])

    def predict_proba(self, X):
        pass

    def get_minority_class(self, y):
        unique_y, counts_y = np.unique(y, return_counts=True)
        return unique_y[np.argmin(counts_y)]
