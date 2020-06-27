from dyn2sel.apply_dcs import DCSApplier
from dyn2sel.ensemble import MDEEnsemble
from dyn2sel.dcs_techniques.desdd_selection import DESDDSel
import numpy as np
from dyn2sel.ensemble import DESDDEnsemble
from skmultiflow.drift_detection import ADWIN


class DESDDMethod(DCSApplier):
    def __init__(self, base_ensemble, drift_detector=ADWIN(), ensemble_size=10, min_lambda=1, max_lambda=6):
        self.ensemble = DESDDEnsemble(
            base_ensemble, ensemble_size=ensemble_size,
            min_lambda=min_lambda, max_lambda=max_lambda)
        self.drift_detector = drift_detector
        self.max_ensemble_size = ensemble_size
        self.min_lambda = min_lambda
        self.max_lambda = max_lambda
        self.dcs_method = DESDDSel()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        self.update_detector(X, y)
        self.ensemble.partial_fit(X, y)

    def predict(self, X):
        if len(self.ensemble) > 0:
            predictions = self.dcs_method.predict(self.ensemble, X)
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
