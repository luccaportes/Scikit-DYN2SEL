from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.ensemble import MDEEnsemble
from dyn2sel.dcs_techniques.mde_selection import MDESel

import numpy as np


class MDEMethod(DCSApplier):
    """
    MDEMethod
    The Minority Driven Ensemble (MDE) is not only a selection method but a whole framework that covers training an
    ensemble with a data stream and predicting new instances. It focuses on data that are affected by the imbalanced
    class problem. This is done by replacing poor performing classifiers with new ones. The method divides the stream
    into chunks of data with a fixed size. Each chunk is passed as a mini-batch for the ensemble to train on. For each
    data chunk, the instances belonging to the minority class are filtered in order to remove outliers. That is done
    using K-Nearest Neighbors on the current data chunk. If the nearest neighbors of each instance belong to the
    majority class, the instance is then considered an outlier and then removed from the chunk. When predicting, if a
    single member predicted the instance as being from the minority class, it is then considered to be from such class.

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
                self.ensemble.partial_fit(
                    np.array(self.temp_buffer_x), np.array(self.temp_buffer_y)
                )
                self.ensemble.classes_ = np.sort(np.unique(self.temp_buffer_y))
                self.minority_class = self.get_minority_class(self.temp_buffer_y)
                self.dcs_method = MDESel(self.minority_class)
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
