from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np
from scipy.stats import mode


class DESDDSel(DCSTechnique):
    def predict(self, ensemble, instances, real_labels=None):
        return ensemble[ensemble.get_max_accuracy()].predict(instances)
