from dyn2sel.dcs_techniques import DCSTechnique
import numpy as np
from scipy.stats import mode


class Oracle(DCSTechnique):
    """
    Oracle
    The idea is to select one of the classifiers of the ensemble (if it exists) that correctly predicted the instance.
    This is obviously not suitable for real problems since it depends on having the class of the instances to predict.
    The sole purpose of this method is to check if there is room available for increasing the performance of the
    ensemble. In other words, if the Oracle considerably outperforms a normal execution, it means that there are
    classifiers that are predicting correctly when the majority is predicting wrongly, thus it is expected that a good
    selection method will bring benefits.
    References
    ----------
        Britto, A. S.; Sabourin, R.; Oliveira, L. E. 2014. Dynamic selection of classifiers —  a comprehensive review.
        Pattern Recognition, v. 47, n. 11, p. 3665 – 3680, 2014. ISSN 0031-3203. Available in:
        <http://www.sciencedirect.com/science/article/pii/S0031320314001885>.
    """

    def predict(self, ensemble, instances, real_labels=None):
        if real_labels is None:
            raise ValueError(
                "Oracle selector needs the real label to run.\n"
                "If you are using an evaluator please use one of the "
                "evaluators in dyn2sel.utils.evaluators"
            )
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

    def _is_oracle(self):
        return True
