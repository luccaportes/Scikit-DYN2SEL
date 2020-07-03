from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np
from scipy.stats import mode


class MDESel(DCSTechnique):
    def __init__(self, minority_class, n_neighbors=7, algorithm="auto"):
        super().__init__(n_neighbors, algorithm)
        self.minority_class = minority_class

    def predict(self, ensemble, instances, real_labels=None):
        predictions = np.empty((instances.shape[0], len(ensemble)))
        for index_clf, clf in enumerate(ensemble):
            predictions[:, index_clf] = clf.predict(instances)
        pred = mode(predictions, axis=1)[0].flatten()
        minority_pred_index = np.any(predictions == self.minority_class, axis=1)
        pred[minority_pred_index] = self.minority_class
        return pred
