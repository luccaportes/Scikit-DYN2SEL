from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.ensemble import DYNSEEnsemble

import numpy as np


class DYNSEMethod(DCSApplier):
    """
    DYNSEMethod
    The Dynamic Selection Based Drift Handler (DYNSE) is a method that applies traditional offline techniques of Dynamic
    Selection in online Machine Learning environments, in order to deal with concept drift. It builds each classifier
    with batches of labeled data that arrive at once. Each batch is used to train a new classifier of the ensemble, and
    if the size of the batch is not sufficiently large, multiple batches can be accumulated before training. Any base
    classifier can be used to compose the ensemble, even the ones intended for offline Machine Learning, since they
    receive the data for training all at once. On the prediction step, when an instance arrives to be predicted, a
    K-Nearest Neighbors search is executed to find the most similar instances to x in the validation set, which is
    defined by the M latest supervised batches that arrived to be trained on. Once the similar instances are
    gathered, any selection method that depends on it can be applied.

    Parameters
    ----------
    clf : Scikit-Multiflow Classifier
        The base classifier used for populating the ensemble

    chunk_size : integer
        The size of the chunks to accumulate data before fitting a classifier.

    dcs_method : DCSTechnique object
        Dynamic selection technique to be used in the prediction process.

    max_ensemble_size : integer, default=-1
        The maximum size that an ensemble can grow. If -1, it grows indefinitely.

    References
    ----------
        Almeida, P. R. L. D.; Oliveira, L. S.; Britto, A. D. S.; Sabourin, R. 2016. Handling concept drifts using
        dynamic selection of classifiers. In:2016 IEEE 28th International Conference on Tools with Artificial
        Intelligence (ICTAI). [S.l.: s.n.]. p. 989–995. ISSN 2375-0197.
    """

    def __init__(self, clf, chunk_size, dcs_method, max_ensemble_size=-1):
        self.clf = clf
        self.chunk_size = chunk_size
        self.max_ensemble_size = max_ensemble_size
        self.dcs_method = dcs_method
        self.ensemble = DYNSEEnsemble(clf)
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
                # self.val_set.replace_set(self.temp_buffer_x, self.temp_buffer_y)
                self.ensemble.classes_ = np.sort(np.unique(self.temp_buffer_y))
                self.dcs_method.fit(
                    np.array(self.temp_buffer_x), np.array(self.temp_buffer_y)
                )
                self.temp_buffer_x = []
                self.temp_buffer_y = []

    def predict(self, X, y=None):
        if len(self.ensemble) > 0:
            predictions = self.dcs_method.predict(self.ensemble, X, y)
            return predictions
        else:
            return np.zeros(X.shape[0])

    def predict_proba(self, X):
        pass
