from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.ensemble import DDCSEnsemble

import numpy as np


class DDCSMethod(DCSApplier):
    """
    DDCSMethod
    The Double Dynamic Selection (DDCS) is a method that applies traditional offline techniques of Dynamic
    Selection in online Machine Learning environments. It builds each classifier with batches of labeled data that
    arrive at once. Each batch is used to train a new classifier of the ensemble and to update all of the other members,
    and if the size of the batch is not sufficiently large, multiple batches can be accumulated before training.
    Any online base classifier can be used to compose the ensemble. On the prediction step, when an instance arrives to
    be predicted, a K-Nearest Neighbors search is executed to find the most similar instances to x in the validation
    set, which is defined by the M latest supervised batches that arrived to be trained on. Once the similar instances
    are gathered, any selection method that depends on it can be applied.

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

    init_all : bool, default=False
        Whether to initialize all of the ensemble at the start of the execution or to populate the ensemble at each
        chunk arrives

    use_bagging : bool, default=False
        Whether to use online bagging when training the base classifiers.


    References
    ----------
        Cavalheiro, L. P; Barddal, J. P; Britto, A. D. S.; Heutte, L. 2021. Dynamically Selected Ensemble for Data
        Stream Classification. In:2021 IEEE International Joint Conference on Neural Networks (IJCNN).
    """

    def __init__(
        self,
        clf,
        chunk_size,
        dcs_method,
        max_ensemble_size=-1,
        init_all=False,
        use_bagging=False,
    ):
        self.clf = clf
        self.chunk_size = chunk_size
        self.max_ensemble_size = max_ensemble_size
        self.dcs_method = dcs_method
        self.init_all = init_all
        self.use_bagging = use_bagging
        self.ensemble = DDCSEnsemble(
            clf, max_size=max_ensemble_size, init_all=init_all, use_bagging=use_bagging
        )
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
        return np.zeros(X.shape[0])

    def predict_proba(self, X):
        pass
