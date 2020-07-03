from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np
from scipy.stats import mode


class Oracle(DCSTechnique):
    def predict(self, ensemble, instances, real_labels=None):
        if real_labels is None:
            raise ValueError("Oracle selector needs the real label to run.\n"
                              "If you are using an evaluator please use one of the "
                              "evaluators in dyn2sel.utils.evaluators")
        preds = np.empty((instances.shape[0], len(ensemble)), dtype=np.int)
        for index_clf, clf in enumerate(ensemble):
            prediction = clf.predict(instances)
            preds[:, index_clf] = prediction
        correct_preds = preds == real_labels[:, None]
        at_least_one = np.any(correct_preds, axis=1)
        final_pred = np.empty_like(real_labels)
        final_pred[at_least_one] = real_labels[at_least_one]
        not_even_one = mode(preds[~at_least_one], axis=1)[0].reshape(1, -1).flatten()
        final_pred[~at_least_one] = not_even_one
        return final_pred

    def is_oracle(self):
        return True



