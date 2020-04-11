from dyn2sel.apply_dcs import DYNSEMethod, WPSMethod
from dyn2sel.dcs_techniques import KNORAE
from skmultiflow.meta import AdaptiveRandomForest, OzaBagging
from skmultiflow.bayes import NaiveBayes
from skmultiflow.trees import HoeffdingTree
from skmultiflow.data import SEAGenerator
from skmultiflow.evaluation import EvaluatePrequential

sea = SEAGenerator()
sea.prepare_for_use()

w = WPSMethod(AdaptiveRandomForest(n_estimators=10), 5, window_size=300, metric="recall")
# w.partial_fit(X_train, y_train)
# w.predict(X_test)

# d = DYNSEMethod(AdaptiveRandomForest(), 1000, KNORAE())
# d = DYNSEMethod(HoeffdingTree(), 1000, KNORAE())
rf = AdaptiveRandomForest()
# #
ev = EvaluatePrequential(
    n_wait=1000, max_samples=10000, pretrain_size=0,
    metrics=["accuracy", "model_size"], output_file="teste.txt")
ev.evaluate(sea, [rf, w], ["ARF", "WPS"])
# ev.evaluate(sea, [rf], [ "ARF"])



# X_train, y_train = sea.next_sample(50)
# X_test, y_test = sea.next_sample(1)

# w.partial_fit(X_train, y_train)
# print(w.predict(X_test))

# X_train, y_train = sea.next_sample(2000)
# X_test, y_test = sea.next_sample(200)
# rf.partial_fit(X_train, y_train)
# print(rf.predict(X_test))

