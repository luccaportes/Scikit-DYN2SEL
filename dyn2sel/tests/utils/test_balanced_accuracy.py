from dyn2sel.utils import BalancedAccuracyEvaluator
from sklearn.metrics import balanced_accuracy_score
import numpy as np


def test_bac():
    y_pred = np.random.randint(0, 10, 5000)
    y_true = np.random.randint(0, 10, 5000)
    bac_eval = BalancedAccuracyEvaluator()
    bac_eval.add_results(y_true, y_pred)
    bac_sklearn = np.round(balanced_accuracy_score(y_true, y_pred), 3)
    bac_evaluator = np.round(bac_eval.get_bac(), 3)
    assert bac_sklearn == bac_evaluator
