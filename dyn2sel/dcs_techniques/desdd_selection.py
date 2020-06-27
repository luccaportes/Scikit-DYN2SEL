from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np
from scipy.stats import mode


class DESDDSel(DCSTechnique):
    def predict(self, ensemble, instances):
        return ensemble[ensemble.get_max_accuracy()].predict(instances)
