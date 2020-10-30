from dyn2sel.apply_dcs import DPDESMethod
from dyn2sel.dcs_techniques import KNORAU
from skmultiflow.data import FileStream
from skmultiflow.evaluation import EvaluatePrequential
from skmultiflow.bayes import NaiveBayes
from skmultiflow.meta import OzaBagging

from sklearn.datasets import make_classification

with open("dataset_imb.csv", "w") as f:
    X, y = make_classification(n_features=10, n_informative=10,
                               n_redundant=0, n_samples=10000,
                               weights=[0.5])
    for i in range(X.shape[0]):
        for att in X[i]:
            f.write(str(att) + ",")
        f.write(str(y[i]) + "\n")


generator = FileStream("dataset_imb.csv")

dynse = DPDESMethod(NaiveBayes(), 200, 10, KNORAU())
ozabag = OzaBagging(NaiveBayes(), n_estimators=10)

evaluator = EvaluatePrequential(max_samples=10000, n_wait=200,
                                batch_size=200, pretrain_size=0, metrics=["precision"])
evaluator.evaluate(generator, [dynse, ozabag], ["Dynse", "Ozabag"])
