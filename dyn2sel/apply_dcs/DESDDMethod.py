from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.ensemble import MDEEnsemble
from dyn2sel.dcs_techniques.desdd_selection import DESDDSel
import numpy as np
from dyn2sel.ensemble import DESDDEnsemble
from skmultiflow.drift_detection import ADWIN


class DESDDMethod(DCSApplier):
    """
    DESDDMethod
    Dynamic Ensemble Selection for Drift Detection (DESDD) method provides a different concept in Dynamic Selection.
    Methods such as KNORA-E and KNORA-U are also referred to as DES methods because they select a subset of the
    ensemble, which is, by definition, also an ensemble. DESDD, in contrast, creates a group of ensembles and selects
    the one with the higher accuracy to be the predictor of the instance.

    Parameters
    ----------
    base_ensemble : Scikit-Multiflow ensemble
        The ensemble used for populating the ensemble of ensembles

    drift_detector : Scikit-Multiflow Drift Detector, default=ADWIN()
        The drift detector used for detecting drift on the ensemble. When a drift is detected the whole ensemble is
        discarded and its construction starts over again.

    ensemble_size : integer, default=10
        The number of ensembles used in the ensemble of ensembles.

    min_lambda : integer, default=1
        The minimum lambda value used for online bagging in the ensemble generation process.

    max_lambda : integer, default=6
        The maximum lambda value used for online bagging in the ensemble generation process.

    References
    ----------
        Albuquerque, R. A. S., Costa, A. F. J., Santos, E. M. dos, Sabourin, R.,Giusti, R.A. 2019. Decision-Based
        Dynamic Ensemble Selection Method for Concept Drift.
    """

    def __init__(
        self,
        base_ensemble,
        drift_detector=ADWIN(),
        ensemble_size=10,
        min_lambda=1,
        max_lambda=6,
    ):
        self.ensemble = DESDDEnsemble(
            base_ensemble,
            ensemble_size=ensemble_size,
            min_lambda=min_lambda,
            max_lambda=max_lambda,
        )
        self.drift_detector = drift_detector
        self.max_ensemble_size = ensemble_size
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.dcs_method = DESDDSel()
        self.classes = None

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.classes is None:
            self.classes = classes
        self.update_detector(X, y)
        self.ensemble.partial_fit(
            X, y, classes=self.classes, sample_weight=sample_weight
        )

    def predict(self, X, y=None):
        if len(self.ensemble) > 0:
            predictions = self.dcs_method.predict(self.ensemble, X, y)
            return predictions
        else:
            return np.array([])

    def predict_proba(self, X):
        pass

    def update_detector(self, X, y):
        if self.drift_detector.detected_change():
            self.ensemble.clear_ensemble()
            self.drift_detector.reset()
        if len(self.ensemble) > 0:
            preds = self.predict(X)
            for i in (preds == y).astype(int):
                self.drift_detector.add_element(i)
